#!/usr/bin/env python3
"""
Quick HRRR/IFS dataset sanity checker.

Validates:
  - 'step' and derived 'lead_time_hrs' (scalar vs 1-D)
  - 'time' and derived 'init_z' (scalar vs 1-D)
  - latitude/longitude detection and 2-D mesh for KDTree
  - (optional) MADIS obs nearest-neighbor search to confirm KDTree works
  - (optional) Topography open_mfdataset with future-safe combine defaults

Usage:
  python check_hrrr_shapes.py --hrrr /path/to/hrrr_or_ifs.nc
  python check_hrrr_shapes.py --hrrr /path/to/hrrr.nc --madis /path/to/madis.nc
  python check_hrrr_shapes.py --hrrr /path/to/hrrr.nc --topo "/path/to/topo/*.nc"
"""

import argparse
import glob
import sys
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import KDTree


def open_and_decode_hrrr(hrrr_fp: str) -> xr.Dataset:
    ds = xr.open_dataset(hrrr_fp, decode_cf=False)
    # Some encodings may include 'dtype' in step attrs that confuse decode_cf
    if "step" in ds and "dtype" in ds["step"].attrs:
        ds["step"].attrs.pop("dtype")
    # Be explicit to avoid timedelta FutureWarning drift
    ds = xr.decode_cf(ds, decode_timedelta=True)
    return ds


def summarize_step(ds: xr.Dataset):
    print("\n[STEP / lead_time_hrs]")
    if "step" not in ds.coords:
        print("  - 'step' coord: MISSING")
        return

    step = ds["step"]
    print(f"  - step.ndim={step.ndim}, dtype={step.dtype}, shape={step.shape}")
    if np.issubdtype(step.dtype, np.timedelta64):
        step_hrs = (step / np.timedelta64(1, "h")).astype("float32")
        print("  - decoded step -> timedelta64 -> hours (float32)")
    else:
        units = str(step.attrs.get("units", "")).lower()
        if "hour" in units or "hr" in units:
            step_hrs = step.astype("float32")
            print("  - numeric step assumed hours (float32)")
        elif "second" in units or units == "s":
            step_hrs = (step.astype("float32") / 3600.0)
            print("  - numeric step converted seconds -> hours (float32)")
        else:
            step_hrs = step.astype("float32")
            print("  - numeric step, unknown units -> treated as hours (float32)")

    if step.ndim == 0:
        print(f"  - lead_time_hrs (scalar): {float(step_hrs.values):.3f} h")
    else:
        vals = np.asarray(step_hrs.values)
        msg = f"  - lead_time_hrs (1-D): len={vals.size}"
        if vals.size <= 10:
            msg += f", values={vals.tolist()}"
        else:
            msg += f", head={vals[:5].tolist()} ..."
        print(msg)


def summarize_time(ds: xr.Dataset):
    print("\n[TIME / init_z]")
    if "time" not in ds.coords:
        print("  - 'time' coord: MISSING")
        return

    time_da = ds["time"]
    print(f"  - time.ndim={time_da.ndim}, dtype={time_da.dtype}, shape={time_da.shape}")

    if time_da.ndim == 0:
        t = pd.to_datetime(time_da.values)
        init_z_val = np.float32(t.hour == 12)
        print(f"  - scalar time={t} -> init_z={init_z_val}")
    else:
        try:
            hours = time_da.dt.hour.values
        except Exception:
            hours = pd.to_datetime(time_da.values).astype("datetime64[h]").astype(object)
            hours = np.array([pd.to_datetime(t).hour for t in hours])
        init_z = (hours == 12).astype("float32")
        msg = f"  - vector time: len={init_z.size}, init_z_sum={init_z.sum()} (count of 12Z)"
        if init_z.size <= 10:
            msg += f", init_z={init_z.tolist()}"
        else:
            msg += f", init_z_head={init_z[:5].tolist()} ..."
        print(msg)


def find_lat_lon(ds: xr.Dataset):
    """
    Try to find latitude and longitude arrays.
    Preference order: coords 'latitude','longitude' -> coords 'lat','lon'
    -> any coord containing 'lat'/'lon' -> data_vars with same names.
    Returns (lat_da, lon_da, source_names) or (None, None, None).
    """
    cand_pairs = [
        ("latitude", "longitude"),
        ("lat", "lon"),
    ]

    # exact coord matches
    for la, lo in cand_pairs:
        if la in ds.coords and lo in ds.coords:
            return ds[la], ds[lo], (la, lo)

    # any coord containing lat/lon
    lat_key = next((k for k in ds.coords if "lat" in k.lower()), None)
    lon_key = next((k for k in ds.coords if "lon" in k.lower()), None)
    if lat_key and lon_key:
        return ds[lat_key], ds[lon_key], (lat_key, lon_key)

    # data_vars fallback
    for la, lo in cand_pairs:
        if la in ds.data_vars and lo in ds.data_vars:
            return ds[la], ds[lo], (la, lo)

    lat_key = next((k for k in ds.data_vars if "lat" in k.lower()), None)
    lon_key = next((k for k in ds.data_vars if "lon" in k.lower()), None)
    if lat_key and lon_key:
        return ds[lat_key], ds[lon_key], (lat_key, lon_key)

    return None, None, None


def latlon_to_mesh(lat_da: xr.DataArray, lon_da: xr.DataArray):
    """
    Return (lat2d, lon2d) as numpy arrays. If inputs are already 2D with equal shape,
    return as-is. If both 1D, meshgrid with indexing='ij'. Otherwise raise.
    """
    lat = lat_da.values
    lon = lon_da.values

    if lat.ndim == 2 and lon.ndim == 2:
        if lat.shape != lon.shape:
            raise ValueError(f"2-D lat/lon shapes differ: {lat.shape} vs {lon.shape}")
        return lat, lon

    if lat.ndim == 1 and lon.ndim == 1:
        lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
        return lat2d, lon2d

    raise ValueError(f"Unsupported lat/lon ranks: lat.ndim={lat.ndim}, lon.ndim={lon.ndim}")


def kdgrid_from_latlon(lat2d: np.ndarray, lon2d: np.ndarray):
    coords = np.column_stack([lat2d.ravel(), lon2d.ravel()])
    print("\n[KDTREE GRID]")
    print(f"  - lat2d.shape={lat2d.shape}, lon2d.shape={lon2d.shape}")
    print(f"  - KDTree coords shape: {coords.shape} (Ncells x 2)")
    tree = KDTree(coords)
    return tree, coords


def load_madis(madis_fp: str):
    ds = xr.open_dataset(madis_fp, engine="netcdf4")
    if "latitude" not in ds or "longitude" not in ds:
        raise KeyError("MADIS file missing 'latitude'/'longitude' variables")
    lat_obs = ds["latitude"].values
    lon_obs = ds["longitude"].values
    print("\n[MADIS]")
    print(f"  - loaded obs: n={lat_obs.size}")
    return lat_obs, lon_obs


def load_topo(glob_pattern: str):
    fps = sorted(glob.glob(glob_pattern))
    if not fps:
        print(f"  - No topo files matched: {glob_pattern}")
        return None
    with xr.set_options(use_new_combine_kwarg_defaults=True):
        dset_topo = xr.open_mfdataset(fps, engine="netcdf4")
    print("\n[TOPO]")
    print(f"  - loaded {len(fps)} files")
    print(f"  - dims: {dset_topo.dims}")
    return dset_topo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hrrr", required=True, help="Path to HRRR/IFS file")
    ap.add_argument("--madis", default=None, help="Optional MADIS file for NN test")
    ap.add_argument("--topo", default=None, help="Optional glob for topo (e.g., '/path/to/topo/*.nc')")
    args = ap.parse_args()

    print(f"[INFO] Opening HRRR/IFS: {args.hrrr}")
    ds = open_and_decode_hrrr(args.hrrr)
    print("\n[DATASET SUMMARY]")
    print(ds)
    print("\n[DIMS]")
    print(ds.dims)
    print("\n[COORDS]")
    print(list(ds.coords))

    # Step / Time summaries
    summarize_step(ds)
    summarize_time(ds)

    # Latitude/Longitude detection and KDTree shape check
    lat_da, lon_da, names = find_lat_lon(ds)
    if lat_da is None or lon_da is None:
        print("\n[ERROR] Could not locate latitude/longitude in dataset.")
        sys.exit(2)
    print(f"\n[LAT/LON DETECTED] using names: {names}")
    print(f"  - lat ndim/shape: {lat_da.ndim}/{lat_da.shape}")
    print(f"  - lon ndim/shape: {lon_da.ndim}/{lon_da.shape}")

    try:
        lat2d, lon2d = latlon_to_mesh(lat_da, lon_da)
    except Exception as e:
        print(f"[ERROR] lat/lon to mesh failed: {e}")
        sys.exit(2)

    tree, coords = kdgrid_from_latlon(lat2d, lon2d)

    # Optional MADIS NN check
    if args.madis:
        lat_obs, lon_obs = load_madis(args.madis)
        coords_obs = np.column_stack([lat_obs.ravel(), lon_obs.ravel()])
        print("\n[KDTREE QUERY with MADIS]")
        dists, inds = tree.query(coords_obs, k=1)
        print(f"  - NN query OK: obs={coords_obs.shape[0]}, "
              f"min_dist={dists.min():.6f}, max_dist={dists.max():.6f}")

    # Optional Topo check
    if args.topo:
        _ = load_topo(args.topo)
        # You can add additional checks here for topo dims/coords if needed.

    print("\n[OK] Sanity checks completed successfully.")


if __name__ == "__main__":
    main()