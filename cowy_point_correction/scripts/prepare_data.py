#!/usr/bin/env python3
"""
Prepare data for CoWy Point Correction:
 - Optionally rsync source data to $TMPDIR
 - Build obs_lookup table (station→grid mapping)
 - Compute feature normalization statistics (mean, std)
 - Save outputs to configured locations

Usage:
    python scripts/prepare_data.py configs/ifs_v1.yaml \
        --terrain-src /project/cowy-nvhackathon/cowy-wildfire/data/terrain_data/terrain_990m \
        --ifs-src /project/cowy-nvhackathon/cowy-wildfire/data/nwp/ifs/
"""

import argparse
import glob
import json
import os
import sys
import subprocess
from datetime import datetime

import numpy as np
import yaml

# Local imports
from cowy.data.dataset import CoWyPointDataset


def _expand_env(s: str) -> str:
    if s is None:
        return s
    return os.path.expandvars(os.path.expanduser(s))


def rsync_dir(src: str, dst: str, includes: list[str] | None = None):
    """
    Rsync a directory tree. If includes supplied, only include patterns.

    Args:
        src: source directory (trailing slash recommended)
        dst: destination directory
        includes: e.g. ['*2024*f72.nc', '*2024*f75.nc']
    """
    os.makedirs(dst, exist_ok=True)

    if not os.path.isdir(src):
        raise FileNotFoundError(f"rsync source not found: {src}")

    cmd = ["rsync", "-avP"]

    # If include filters are given, use --include/--exclude
    if includes:
        for inc in includes:
            cmd.extend(["--include", inc])
        cmd.extend(["--exclude", "*"])

    cmd.extend([src.rstrip("/") + "/", dst.rstrip("/") + "/"])
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def compute_stats(dataset: CoWyPointDataset, save_mean: str, save_std: str):
    """
    Iterate the dataset once to compute mean/std of feature vectors.
    Saves arrays to disk (np.save).
    """
    X = []
    Y = []

    n = len(dataset)
    print(f"Computing feature statistics over {n:,} samples (this may take a while)…")

    for i in range(n):
        x, y = dataset[i]
        X.append(x)
        Y.append(y)

        if (i + 1) % 100000 == 0:
            print(f"  processed {i+1:,}/{n:,}")

    X = np.vstack(X).astype(np.float32)      # (n_samples, n_features)
    Y = np.vstack(Y).astype(np.float32)      # could be (n_samples, 1)

    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)

    os.makedirs(os.path.dirname(save_mean), exist_ok=True)
    np.save(save_mean, mean)
    np.save(save_std, std)

    print(f"Saved mean to: {save_mean}")
    print(f"Saved std  to: {save_std}")

    return mean, std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config")
    parser.add_argument("--terrain-src", default=None,
                        help="Source dir for terrain rasters (will rsync to cfg.paths.terrain_dir)")
    parser.add_argument("--ifs-src", default=None,
                        help="Source dir for IFS/HRRR netCDFs (will rsync to cfg.paths.ifs_dir)")
    parser.add_argument("--ifs-include", nargs="*", default=[
        "*2024*f72.nc", "*2024*f75.nc", "*2024*f78.nc",
        "*2024*f81.nc", "*2024*f84.nc", "*2024*f87.nc",
        "*2024*f90.nc", "*2024*f93.nc", "*2024*f96.nc"
    ], help="Patterns to include when syncing IFS/HRRR files")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    madis_fp = _expand_env(paths["madis"])

    tmpdir = os.environ.get("TMPDIR", "/tmp")
    print(f"Using TMPDIR: {tmpdir}")

    terrain_dir = _expand_env(paths.get("terrain_dir", os.path.join(tmpdir, "terrain")))
    ifs_dir = _expand_env(paths.get("ifs_dir", os.path.join(tmpdir, "ifs")))

    # Optionally rsync to TMPDIR
    if args.terrain_src:
        rsync_dir(_expand_env(args.terrain_src), terrain_dir)
    else:
        print("Skipping terrain rsync (no --terrain-src provided); assuming data present.")

    if args.ifs_src:
        rsync_dir(_expand_env(args.ifs_src), ifs_dir, includes=args.ifs_include)
    else:
        print("Skipping IFS/HRRR rsync (no --ifs-src provided); assuming data present.")

    # Collect files
    topo_fps = sorted(glob.glob(os.path.join(terrain_dir, "*_reprojected_wgs84_cowy_990m.nc")))
    ifs_fps = sorted(glob.glob(os.path.join(ifs_dir, "*.nc")))

    if len(topo_fps) == 0:
        print(f"WARNING: No topo files found in {terrain_dir}")
    if len(ifs_fps) == 0:
        print(f"WARNING: No IFS/HRRR files found in {ifs_dir}")

    # Build dataset and obs_lookup
    print("Building CoWyPointDataset for metadata (obs_lookup)…")
    dset = CoWyPointDataset(
        madis_fp=madis_fp,
        hrrr_fps=ifs_fps,
        topo_fps=topo_fps,
        dist_lim=cfg["data"]["dist_lim"],
    )

    # Save obs_lookup
    obs_lookup_fp = _expand_env(paths["obs_lookup"])
    os.makedirs(os.path.dirname(obs_lookup_fp), exist_ok=True)
    dset.obs_lookup.to_csv(obs_lookup_fp, index=False)
    print(f"Saved obs_lookup: {obs_lookup_fp}")

    # Compute stats and save
    mean_fp = _expand_env(paths["mean_file"])
    std_fp = _expand_env(paths["std_file"])
    compute_stats(dset, mean_fp, std_fp)

    # Save a small prep log
    out_root = _expand_env(paths["output_root"])
    os.makedirs(out_root, exist_ok=True)
    meta = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "n_obs": len(dset),
        "ifs_files": len(ifs_fps),
        "topo_files": len(topo_fps),
        "obs_lookup": obs_lookup_fp,
        "mean": mean_fp,
        "std": std_fp,
    }
    with open(os.path.join(out_root, "prepare_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Preparation complete.")


if __name__ == "__main__":
    sys.exit(main())

