
# cowy/data/dataset.py
import os
import copy
import random
import numpy as np
import pandas as pd
import xarray as xr
from torch.utils.data import Dataset
from scipy.spatial import KDTree

from cowy.physics.lapse_rate import compute_elr


def add_forecast_metadata(ds: xr.Dataset) -> xr.Dataset:
    """
    Add forecast metadata to an HRRR/IFS xarray Dataset:
      - lead_time_hrs: derived from 'step' (as coordinate aligned to 'step')
      - init_z: 1.0 if init hour == 12Z else 0.0, aligned to 'time'
    Notes:
      * This function is safe if 'step' or 'time' are absent.
      * If files contain multiple 'time' entries, you may want to align or select
        a specific index before caching/stacking downstream.
    """
    ds = ds.copy()

    # lead_time_hrs along 'step'
    if "step" in ds.coords:
        if np.issubdtype(ds["step"].dtype, np.timedelta64):
            lead_time_hrs = (ds["step"] / np.timedelta64(1, "h")).astype("float32")
        else:
            # if already numeric, cast to float32
            lead_time_hrs = ds["step"].astype("float32")
        ds = ds.assign_coords(lead_time_hrs=("step", lead_time_hrs))

    # init_z along 'time'
    if "time" in ds.coords:
        times = pd.to_datetime(ds["time"].values)
        init_z = np.array([(t.hour == 12) for t in times], dtype="float32")
        ds["init_z"] = xr.DataArray(init_z, dims=["time"])

    return ds


def add_derived_hrrr_variables(ds: xr.Dataset) -> xr.Dataset:
    """
    Add derived HRRR/IFS variables that are used as features:
      - ws_10m = sqrt(u10^2 + v10^2)
      - elr via compute_elr(ds)
    This function assumes compute_elr(ds) returns an xarray.DataArray.
    """
    ds = ds.copy()

    # 10 m wind speed
    if {"u10", "v10"}.issubset(ds.data_vars):
        ds["ws_10m"] = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)

    # Environmental lapse rate (compute_elr determines inputs internally)
    # Ensure compute_elr returns DataArray with broadcastable dims
    try:
        elr_da = compute_elr(ds)
        ds["elr"] = elr_da
    except Exception as e:
        # Optional: log or raise depending on your preference
        # For robustness, we don't fail hard here; comment out raise to keep permissive
        # raise
        pass

    return ds


class CoWyPointDataset(Dataset):
    """
    Point-based dataset mapping MADIS obs to nearest HRRR/IFS gridpoint,
    including terrain covariates. Logic mirrors the original notebook.

    This version augments HRRR datasets with:
      - lead_time_hrs (coord along 'step')
      - init_z (variable along 'time')
      - ws_10m (derived from u10, v10)
      - elr (from cowy.physics.lapse_rate.compute_elr)
    """

    def __init__(
        self,
        madis_fp: str,
        hrrr_fps: list[str],
        topo_fps: list[str],
        dist_lim: float = 0.25,
        shuffle: bool = True,
        obs_meta_cache: str | None = None,
    ):
        self.dist_lim = dist_lim
        self.shuffle = shuffle

        # --- MADIS ---
        self.dset_madis = xr.open_dataset(madis_fp, engine="h5netcdf")[["windspeed_10m"]]
        self.vars_madis = list(self.dset_madis.data_vars)
        self.lat_obs = self.dset_madis.latitude.values
        self.lon_obs = self.dset_madis.longitude.values
        self.ti_madis = pd.to_datetime(self.dset_madis.time.values)

        self.madis_arrays = {v: self.dset_madis[v].values for v in self.vars_madis}

        # --- HRRR / IFS ---
        self.dsets_hrrr = {}
        for fp in hrrr_fps:
            ds = xr.open_dataset(fp, decode_cf=False)

            # Some HRRR encodings include step attrs that break decode_cf
            if "step" in ds and "dtype" in ds["step"].attrs:
                ds["step"].attrs.pop("dtype")
            ds = xr.decode_cf(ds)

            # --- Add forecast metadata and derived variables here ---
            ds = add_forecast_metadata(ds)
            ds = add_derived_hrrr_variables(ds)

            # If multiple time values exist in a single file and downstream assumes
            # scalar-time files, you *may* want to select time=0. Uncomment if needed:
            # if "time" in ds.dims and ds.dims["time"] > 1:
            #     ds = ds.isel(time=0)

            # 2D lat/lon coords
            # Input files often have 1D lat/lon; convert to 2D (y,x) for KDTree indexing
            lat1d = ds.latitude.values
            lon1d = ds.longitude.values
            lat2d, lon2d = np.meshgrid(lat1d, lon1d, indexing="ij")
            ds = ds.assign_coords(
                latitude=(("y", "x"), lat2d),
                longitude=(("y", "x"), lon2d)
            )

            self.dsets_hrrr[fp] = ds

        # derive variable list AFTER augmentation, preserves insertion order
        sample_hrrr = next(iter(self.dsets_hrrr.values()))
        self.vars_hrrr = list(sample_hrrr.data_vars)

        # valid_time → filepath
        self.times_hrrr = {
            pd.to_datetime(ds.valid_time.values): fp
            for fp, ds in self.dsets_hrrr.items()
        }

        # filter MADIS by forecast times
        valid_times = np.array(list(self.times_hrrr.keys()), dtype="datetime64[ns]")
        mask = np.isin(self.dset_madis.time.values.astype("datetime64[ns]"), valid_times)
        self.dset_madis = self.dset_madis.isel(time=mask)
        self.ti_madis = pd.to_datetime(self.dset_madis.time.values)

        # --- TOPO ---
        self.dset_topo = xr.open_mfdataset(topo_fps, engine="h5netcdf")
        self.vars_topo = [v for v in self.dset_topo.data_vars if self.dset_topo[v].ndim == 2]

        lat1d_topo = self.dset_topo.latitude.values
        lon1d_topo = self.dset_topo.longitude.values
        self.lon_topo, self.lat_topo = np.meshgrid(lon1d_topo, lat1d_topo)
        self.shape_topo = self.lat_topo.shape

        # --- KDtrees ---
        hrrr_lat = sample_hrrr.latitude.values
        hrrr_lon = sample_hrrr.longitude.values

        coords_hrrr = np.column_stack([hrrr_lat.ravel(), hrrr_lon.ravel()])
        coords_obs = np.column_stack([self.lat_obs, self.lon_obs])
        tree_hrrr = KDTree(coords_hrrr)
        self.nn_dist, self.nn_ind = tree_hrrr.query(coords_obs)

        self.size_hrrr = hrrr_lat.size
        self.shape_hrrr = hrrr_lat.shape
        ind_grid = np.arange(self.size_hrrr).reshape(self.shape_hrrr)

        topo_coords = np.column_stack([self.lat_topo.ravel(), self.lon_topo.ravel()])
        tree_topo = KDTree(topo_coords)
        _, topo_ind = tree_topo.query(coords_obs)
        self.idy_topo, self.idx_topo = np.unravel_index(topo_ind, self.shape_topo)

        idy = xr.DataArray(self.idy_topo.astype(np.int32), dims="nobs")
        idx = xr.DataArray(self.idx_topo.astype(np.int32), dims="nobs")

        topo_vals = []
        for v in self.vars_topo:
            topo_vals.append(self.dset_topo[v].isel(latitude=idy, longitude=idx))
        self.topo_per_obs = xr.concat(topo_vals, dim="var").compute().values.T.astype(np.float32)

        self.idx_obs_inbounds = np.where(self.nn_dist < self.dist_lim)[0]

        # --- obs_lookup ---
        if obs_meta_cache is None:
            self._set_obs_lookup(ind_grid)
        else:
            self.obs_lookup = pd.read_csv(obs_meta_cache)
            self.obs_lookup["timestamp"] = pd.to_datetime(self.obs_lookup["timestamp"])

        self.obs_lookup["obs_number"] = np.arange(len(self.obs_lookup))

        # caching
        self.cache_timestamp = None
        self.cache_hrrr = None
        self.cache_madis = None

    def _set_obs_lookup(self, ind_grid):
        data = {
            "idx_obs": [],
            "idt_madis": [],
            "timestamp": [],
            "fp_hrrr": [],
            "idy_hrrr": [],
            "idx_hrrr": [],
            "idy_topo": [],
            "idx_topo": [],
            "latitude": [],
            "longitude": [],
        }

        for idt, ts in enumerate(self.ti_madis):
            if ts not in self.times_hrrr:
                continue

            obs_arr = self.madis_arrays["windspeed_10m"][idt]
            idx_valid = np.where(~np.isnan(obs_arr))[0]
            idx_valid = sorted(set(idx_valid).intersection(self.idx_obs_inbounds))

            if not idx_valid:
                continue

            fp = self.times_hrrr[ts]
            for i in idx_valid:
                y, x = np.where(ind_grid == self.nn_ind[i])
                data["idx_obs"].append(i)
                data["idt_madis"].append(idt)
                data["timestamp"].append(ts)
                data["fp_hrrr"].append(fp)
                data["idy_hrrr"].append(int(y[0]))
                data["idx_hrrr"].append(int(x[0]))
                data["idy_topo"].append(int(self.idy_topo[i]))
                data["idx_topo"].append(int(self.idx_topo[i]))
                data["latitude"].append(self.lat_obs[i])
                data["longitude"].append(self.lon_obs[i])

        self.obs_lookup = pd.DataFrame(data)

    def shuffle_timesteps(self):
        ts = self.obs_lookup.timestamp.unique()
        order = pd.Series(np.random.permutation(ts), index=ts)
        self.obs_lookup["_ord"] = self.obs_lookup.timestamp.map(order)
        self.obs_lookup = self.obs_lookup.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)

    def _get_cached(self, idt_madis, timestamp, fp):
        ds = self.dsets_hrrr[fp]
        if self.cache_timestamp != timestamp:
            self.cache_timestamp = timestamp
            # Stack HRRR variables according to vars_hrrr insertion order, now including ws_10m/elr/init_z
            self.cache_hrrr = np.dstack([ds[v].values for v in self.vars_hrrr]).astype(np.float32)
            self.cache_madis = np.vstack([
                self.madis_arrays[v][idt_madis] for v in self.vars_madis
            ]).T.astype(np.float32)
        return self.cache_hrrr, self.cache_madis

    def __len__(self):
        return len(self.obs_lookup)

    def __getitem__(self, idx):
        if idx == 0 and self.shuffle:
            self.shuffle_timesteps()

        r = self.obs_lookup.iloc[idx]
        hrrr_arr, madis_arr = self._get_cached(
            r.idt_madis, r.timestamp, r.fp_hrrr
        )

        hrrr_vals = hrrr_arr[int(r.idy_hrrr), int(r.idx_hrrr)]
        topo_vals = self.topo_per_obs[int(r.idx_obs)]
        x = np.concatenate([hrrr_vals, topo_vals]).astype(np.float32)
        y = madis_arr[int(r.idx_obs)]

        if np.isnan(y).any():
            return self[(idx + 1) % len(self)]

        return x, y

# EOF