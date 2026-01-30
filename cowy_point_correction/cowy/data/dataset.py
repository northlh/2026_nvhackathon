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
      - lead_time_hrs: derived from 'step' (scalar if step is scalar, else coord along 'step')
      - init_z: 1.0 if init hour == 12Z else 0.0 (scalar if time is scalar, else vector along 'time')
    """
    ds = ds.copy()

    # --- lead_time_hrs from 'step' ---
    if "step" in ds.coords:
        step = ds["step"]

        # Normalize to hours (float32), regardless of representation
        if np.issubdtype(step.dtype, np.timedelta64):
            step_hrs = (step / np.timedelta64(1, "h")).astype("float32")
        else:
            units = str(step.attrs.get("units", "")).lower()
            if "hour" in units or "hr" in units:
                step_hrs = step.astype("float32")
            elif "second" in units or units == "s":
                step_hrs = (step.astype("float32") / 3600.0)
            else:
                step_hrs = step.astype("float32")

        # Scalar vs 1-D step
        if step.ndim == 0:
            ds = ds.assign_coords(lead_time_hrs=float(step_hrs.values))
        else:
            ds = ds.assign_coords(lead_time_hrs=("step", np.asarray(step_hrs.values)))

    # --- init_z along 'time' (or scalar if 'time' is scalar) ---
    if "time" in ds.coords:
        time_da = ds["time"]
        if time_da.ndim == 0:
            t = pd.to_datetime(time_da.values)
            init_z_val = np.float32(t.hour == 12)
            ds["init_z"] = xr.DataArray(init_z_val)
        else:
            try:
                init_z = (time_da.dt.hour == 12).astype("float32")
            except Exception:
                times = pd.to_datetime(time_da.values)
                init_z = np.array([(t.hour == 12) for t in times], dtype="float32")
            ds["init_z"] = xr.DataArray(init_z, dims=["time"])

    return ds


def add_derived_hrrr_variables(ds: xr.Dataset) -> xr.Dataset:
    """
    Add derived HRRR/IFS variables used as features:
      - ws_10m = sqrt(u10^2 + v10^2)
      - elr via compute_elr(ds)
    """
    ds = ds.copy()

    if {"u10", "v10"}.issubset(ds.data_vars):
        ds["ws_10m"] = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)

    try:
        elr_da = compute_elr(ds)
        ds["elr"] = elr_da
    except Exception:
        # If ELR cannot be computed due to missing inputs, skip silently
        pass

    return ds


class CoWyPointDataset(Dataset):
    """
    Point-based dataset mapping MADIS obs to nearest HRRR/IFS gridpoint,
    including terrain covariates.

    Augments HRRR datasets with:
      - lead_time_hrs (coord along 'step' if vector; scalar otherwise)
      - init_z (variable along 'time' if vector; scalar otherwise)
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
        self.dset_madis = xr.open_dataset(madis_fp, engine="netcdf4")[["windspeed_10m"]]
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

            ds = xr.decode_cf(ds, decode_timedelta=True)

            ds = add_forecast_metadata(ds)
            ds = add_derived_hrrr_variables(ds)

            # 2D lat/lon coords: promote 1-D rectilinear to 2-D on horizontal dims
            if (
                "latitude" in ds.coords
                and "longitude" in ds.coords
                and ds["latitude"].ndim == 1
                and ds["longitude"].ndim == 1
            ):
                lat_da = ds["latitude"]
                lon_da = ds["longitude"]
                dims_2d = None

                # Prefer ('y','x') if present and sizes match
                if ("y" in ds.sizes and "x" in ds.sizes
                        and ds.sizes["y"] == lat_da.size
                        and ds.sizes["x"] == lon_da.size):
                    dims_2d = ("y", "x")
                else:
                    # Infer from any data var with consecutive dims matching (lat_len, lon_len)
                    for v in ds.data_vars:
                        dv = ds[v]
                        if dv.ndim >= 2:
                            for i in range(dv.ndim - 1):
                                d0, d1 = dv.dims[i], dv.dims[i + 1]
                                if ds.sizes.get(d0) == lat_da.size and ds.sizes.get(d1) == lon_da.size:
                                    dims_2d = (d0, d1)
                                    break
                        if dims_2d:
                            break

                if dims_2d:
                    lat2d, lon2d = np.meshgrid(lat_da.values, lon_da.values, indexing="ij")
                    ds = ds.assign_coords(
                        latitude=(dims_2d, lat2d),
                        longitude=(dims_2d, lon2d)
                    )

            self.dsets_hrrr[fp] = ds

        # derive variable list AFTER augmentation
        sample_hrrr = next(iter(self.dsets_hrrr.values()))
        self.vars_hrrr = list(sample_hrrr.data_vars)

        # valid_time → filepath (assumes scalar valid_time per file)
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
        with xr.set_options(use_new_combine_kwarg_defaults=True):
            self.dset_topo = xr.open_mfdataset(topo_fps, engine="netcdf4")
        self.vars_topo = [v for v in self.dset_topo.data_vars if self.dset_topo[v].ndim == 2]

        lat1d_topo = self.dset_topo.latitude.values
        lon1d_topo = self.dset_topo.longitude.values
        self.lon_topo, self.lat_topo = np.meshgrid(lon1d_topo, lat1d_topo)
        self.shape_topo = self.lat_topo.shape

        # --- KDtrees ---
        hrrr_lat = sample_hrrr.latitude.values
        hrrr_lon = sample_hrrr.longitude.values

        # If lat/lon are 1-D axes, build a 2-D mesh for KDTree
        if hrrr_lat.ndim == 1 and hrrr_lon.ndim == 1:
            hrrr_lat, hrrr_lon = np.meshgrid(hrrr_lat, hrrr_lon, indexing="ij")

        coords_hrrr = np.column_stack([hrrr_lat.ravel(), hrrr_lon.ravel()])
        coords_obs = np.column_stack([self.lat_obs, self.lon_obs])
        tree_hrrr = KDTree(coords_hrrr)
        self.nn_dist, self.nn_ind = tree_hrrr.query(coords_obs)

        self.size_hrrr = hrrr_lat.size
        self.shape_hrrr = hrrr_lat.shape

        # Precompute (row, col) indices for each observation's nearest HRRR cell
        # This replaces the old O(N) np.where() scan with O(1) lookups.
        self.idy_hrrr_all, self.idx_hrrr_all = np.unravel_index(self.nn_ind, self.shape_hrrr)

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
            self._set_obs_lookup()
        else:
            self.obs_lookup = pd.read_csv(obs_meta_cache)
            self.obs_lookup["timestamp"] = pd.to_datetime(self.obs_lookup["timestamp"])

        self.obs_lookup["obs_number"] = np.arange(len(self.obs_lookup))

        # caching
        self.cache_timestamp = None
        self.cache_hrrr = None
        self.cache_madis = None

    def _set_obs_lookup(self):
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
                data["idx_obs"].append(i)
                data["idt_madis"].append(idt)
                data["timestamp"].append(ts)
                data["fp_hrrr"].append(fp)
                # Fast O(1) lookup using precomputed unravel indices
                data["idy_hrrr"].append(int(self.idy_hrrr_all[i]))
                data["idx_hrrr"].append(int(self.idx_hrrr_all[i]))
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
        """
        Build (and cache) the stacked HRRR/IFS tensor for the given timestamp/file and the
        MADIS observation array for the aligned time index.

        Robust against:
          - variables with extra dims (time/step) -> slices first entry
          - scalar/1D variables (e.g., init_z) -> broadcasts to 2D grid
          - missing variables -> fills with 2D NaNs
          - dims order differences -> transposes to grid dims
        """
        ds = self.dsets_hrrr[fp]
        if self.cache_timestamp == timestamp:
            return self.cache_hrrr, self.cache_madis

        # --- Determine a reference 2D grid (dims + shape) ---
        time_like = ("time", "valid_time", "forecast_time", "step", "lead_time", "lead_time_hrs")
        ref = None
        grid_dims = None
        grid_shape = None

        # Prefer a data_var with (latitude, longitude) or (y, x) after squeezing out time-like dims
        for v in ds.data_vars:
            a = ds[v]
            for td in time_like:
                if td in a.dims:
                    a = a.isel({td: 0})
            a = a.squeeze(drop=True)
            if set(a.dims) >= {"latitude", "longitude"}:
                a = a.transpose("latitude", "longitude")
                ref = xr.zeros_like(a, dtype=np.float32)
                grid_dims = ref.dims
                grid_shape = ref.shape
                break
            if set(a.dims) >= {"y", "x"}:
                a = a.transpose("y", "x")
                ref = xr.zeros_like(a, dtype=np.float32)
                grid_dims = ref.dims
                grid_shape = ref.shape
                break

        # Fallback: construct grid from known sizes if coords are 1D
        if ref is None:
            if "latitude" in ds.sizes and "longitude" in ds.sizes:
                grid_dims = ("latitude", "longitude")
                grid_shape = (ds.sizes["latitude"], ds.sizes["longitude"])
                ref = xr.DataArray(np.zeros(grid_shape, dtype=np.float32), dims=grid_dims)
            elif "y" in ds.sizes and "x" in ds.sizes:
                grid_dims = ("y", "x")
                grid_shape = (ds.sizes["y"], ds.sizes["x"])
                ref = xr.DataArray(np.zeros(grid_shape, dtype=np.float32), dims=grid_dims)
            else:
                raise ValueError("Could not infer a 2D horizontal grid from dataset variables or sizes.")

        # --- Build robust stack across requested variables ---
        stack_list = []
        for v in self.vars_hrrr:
            if v not in ds:
                # Missing variable for this file -> fill with NaNs of grid shape
                stack_list.append(np.full(grid_shape, np.nan, dtype=np.float32))
                continue

            arr = ds[v]
            # Reduce time-like dims
            for td in time_like:
                if td in arr.dims:
                    arr = arr.isel({td: 0})
            arr = arr.squeeze(drop=True)

            # Ensure a 2D grid-aligned DataArray
            if set(grid_dims).issubset(set(arr.dims)):
                # Reorder to match grid dims
                try:
                    arr = arr.transpose(*grid_dims)
                except Exception:
                    # Try broadcasting then transpose
                    arr = arr.broadcast_like(ref).transpose(*grid_dims)
            else:
                # Scalar or 1D -> broadcast to grid
                if not isinstance(arr, xr.DataArray):
                    arr = xr.DataArray(arr)
                arr = arr.broadcast_like(ref)
                if tuple(arr.dims) != grid_dims:
                    arr = arr.transpose(*grid_dims)

            # Final enforcement of shape
            if arr.shape != grid_shape:
                arr = arr.broadcast_like(ref)
                if tuple(arr.dims) != grid_dims:
                    arr = arr.transpose(*grid_dims)

            stack_list.append(arr.values.astype(np.float32))

        self.cache_hrrr = np.dstack(stack_list).astype(np.float32)

        # --- MADIS cache for the corresponding observation timestep ---
        self.cache_madis = np.vstack(
            [self.madis_arrays[v][idt_madis] for v in self.vars_madis]
        ).T.astype(np.float32)

        self.cache_timestamp = timestamp
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