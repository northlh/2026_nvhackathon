"""
CoWy DataModule

Usage (two-step workflow):
1) Prepare once (slow; writes split key CSVs under paths.obs_lookup's directory):
   python train.py config.yaml --prepare-only

2) Train (fast startup; reuses prepared split keys):
   python train.py config.yaml
"""

import os
import glob
import copy
import random
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightning as L
import torch
from torch.utils.data import DataLoader
import warnings

from cowy.data.dataset import CoWyPointDataset
from cowy.data.spatial import spatial_blocking


# ---------------------------
# Helpers
# ---------------------------

def _expand_env(s: str) -> str:
    """Expand environment variables and ~ in paths."""
    return os.path.expandvars(os.path.expanduser(s))


def _seed_worker(worker_id: int):
    """
    Seed numpy & python's RNG per worker for reproducibility.
    Lightning sets a base seed per process; we derive a worker seed from it.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _ensure_zero_based_idx_obs(idx_obs: np.ndarray, n_obs: int) -> Tuple[np.ndarray, bool]:
    """
    Ensure idx_obs is zero-based and within [0, n_obs-1].
    If it looks 1-based (min>=1 and max==n_obs), shift by -1.

    Returns:
        (idx_obs_normalized, was_shifted)

    Raises:
        ValueError if still out of bounds after normalization (likely stale obs_lookup or mismatch).
    """
    idx_obs = np.asarray(idx_obs).astype(np.int64, copy=False)
    was_shifted = False
    if idx_obs.size == 0:
        return idx_obs, was_shifted

    min_i = int(idx_obs.min())
    max_i = int(idx_obs.max())

    # Heuristic: 1-based if min>=1 and max==n_obs (e.g., 1..N)
    if min_i >= 1 and max_i == n_obs:
        idx_obs = idx_obs - 1
        was_shifted = True
        min_i -= 1
        max_i -= 1

    if min_i < 0 or max_i >= n_obs:
        raise ValueError(
            "idx_obs out of bounds after normalization: "
            f"min={min_i}, max={max_i}, allowed=[0, {n_obs-1}]. "
            "This usually indicates stale obs_lookup.csv or mixed 1-based vs 0-based indexing."
        )

    return idx_obs, was_shifted


def _balance_train_classes(train_ds, thresholds=[5, 11], seed: int = 42):
    """
    Upsample bins to the size of the largest bin.
    Operates by rewriting train_ds.obs_lookup only (datasets/memory-mapped arrays unchanged).
    """
    print("Balancing training classes …")
    df = train_ds.obs_lookup.copy()

    # Pull the observation values corresponding to obs_lookup rows
    obs_all = train_ds.dset_madis["windspeed_10m"].values  # shape: [time, station]
    obs_all_np = np.asarray(obs_all)

    # Normalize and validate indices before advanced indexing
    idt = df["idt_madis"].to_numpy(dtype=np.int64, copy=False)
    idx_obs_raw = df["idx_obs"].to_numpy()
    n_obs = int(obs_all_np.shape[1])

    idx_obs, shifted = _ensure_zero_based_idx_obs(idx_obs_raw, n_obs)
    if shifted:
        warnings.warn(
            "Detected 1-based 'idx_obs' in obs_lookup; converting to 0-based for indexing.",
            RuntimeWarning,
        )

    # Safe advanced indexing; obs is 1D aligned with df rows
    obs = obs_all_np[idt, idx_obs]

    thresholds = sorted(thresholds)
    edges = [0] + thresholds + [np.inf]

    # Partition obs_lookup into bins by observed wind speed
    parts = []
    sizes = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (obs >= lo) & (obs < hi)
        part = df.loc[mask]
        parts.append(part)
        sizes.append(int(mask.sum()))

    max_size = max(sizes) if sizes else 0
    print(f"Bin sizes: {sizes} → target (upsampled) size per bin: {max_size}")

    rng = np.random.RandomState(seed)
    balanced = []
    for part, size in zip(parts, sizes):
        if size == 0:
            continue
        if size < max_size:
            up = part.sample(n=max_size, replace=True, random_state=rng)
            balanced.append(up)
        else:
            balanced.append(part)

    if balanced:
        out = pd.concat(balanced, ignore_index=True)
        out = out.sample(frac=1.0, random_state=rng).reset_index(drop=True)
        train_ds.obs_lookup = out
        print(f"Balanced training size: {len(train_ds)}")
    else:
        train_ds.obs_lookup = df.reset_index(drop=True)
        print("No non-empty bins; training set unchanged.")

    return train_ds


# ---------------------------
# DataModule
# ---------------------------

class CoWyDataModule(L.LightningDataModule):
    """
    Lightning DataModule with a one-time 'prepare()' to cache split keys
    and a fast 'setup()' that reuses them.

    YAML keys under `data:` that this module reads:
      - num_workers: int (default 0)
      - pin_memory: bool (default True)
      - persistent_workers: bool (default False; effective only if num_workers > 0)
      - prefetch_factor: int|None (effective only if num_workers > 0)
      - drop_last: bool (default False)
      - pin_memory_device: str|None (e.g., "cuda", only for recent PyTorch versions)
      - multiprocessing_context: str|None ("fork", "spawn", "forkserver") — optional
      - val_fraction: float
      - balance_classes: bool
      - balance_thresholds: list[int]
      - max_hrrr_files: int (optional; limit files for debugging)

    Notes:
      * We keep shuffle=False in DataLoaders because your dataset handles timestep-aware shuffling internally.
      * Batch size is sourced from cfg["training"]["batch_size"].
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

        # Keep batch size in the training section (as in your setup)
        self.batch_size = int(cfg["training"]["batch_size"])

        # DataLoader config (with safe defaults)
        data_cfg = cfg.get("data", {})
        self.dl_num_workers = int(data_cfg.get("num_workers", 0))
        self.dl_pin_memory = bool(data_cfg.get("pin_memory", True))
        self.dl_persistent_workers = bool(data_cfg.get("persistent_workers", False))
        self.dl_prefetch_factor = data_cfg.get("prefetch_factor", None)
        self.dl_drop_last = bool(data_cfg.get("drop_last", False))
        self.dl_pin_memory_device = data_cfg.get("pin_memory_device", None)
        self.dl_mp_context = data_cfg.get("multiprocessing_context", None)

        self.val_fraction = float(data_cfg["val_fraction"])

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        # Optional: generator for reproducible sampling (works with worker_init_fn)
        self._dl_generator = torch.Generator()
        seed = int(cfg["experiment"]["seed"])
        self._dl_generator.manual_seed(seed)

        # Derive file paths for prepared split key CSVs from obs_lookup stem
        self._split_paths = self._derive_split_paths()

    # ---------------------------
    # Preparation helpers
    # ---------------------------

    def _derive_split_paths(self) -> Dict[str, str]:
        """
        Derive train/val/test key CSVs from cfg.paths.obs_lookup (used as a stem).
        Example:
          paths.obs_lookup: /.../prepare_data_out/obs_lookup.csv
          → /.../prepare_data_out/obs_lookup_train.csv
            /.../prepare_data_out/obs_lookup_val.csv
            /.../prepare_data_out/obs_lookup_test.csv
        """
        stem_path = Path(_expand_env(self.cfg["paths"]["obs_lookup"]))
        parent = stem_path.parent
        stem = stem_path.stem  # e.g., 'obs_lookup'
        return {
            "train": str(parent / f"{stem}_train.csv"),
            "val":   str(parent / f"{stem}_val.csv"),
            "test":  str(parent / f"{stem}_test.csv"),
        }

    def _prepared_exists(self) -> bool:
        p = self._split_paths
        return all(os.path.exists(p[k]) for k in ("train", "val", "test"))

    # ---------------------------
    # NEW: One-time prepare step
    # ---------------------------

    def prepare(self):
        """
        One-time expensive operations:
          - Build base dataset
          - Spatial blocking → train_full and test
          - Train/val split on train_full
          - Save keys (timestamp, idx_obs) per split to CSV
        """
        print("Preparing data (indexing, spatial blocking, splits)…")

        paths = self.cfg["paths"]
        data_cfg = self.cfg["data"]
        seed = int(self.cfg["experiment"]["seed"])

        madis_fp = _expand_env(paths["madis"])
        terrain_dir = _expand_env(paths["terrain_dir"])
        ifs_dir = _expand_env(paths["ifs_dir"])

        topo_fps = sorted(glob.glob(os.path.join(terrain_dir, "*_reprojected_wgs84_cowy_990m.nc")))
        hrrr_fps = sorted(glob.glob(os.path.join(ifs_dir, "*.nc")))
        max_files = data_cfg.get("max_hrrr_files", None)
        if max_files is not None:
            hrrr_fps = hrrr_fps[:int(max_files)]

        # Build base dataset & run spatial blocking (can be slow)
        base_ds = CoWyPointDataset(
            madis_fp=madis_fp,
            hrrr_fps=hrrr_fps,
            topo_fps=topo_fps,
            dist_lim=data_cfg["dist_lim"],
        )
        train_full_ds, test_ds = spatial_blocking(
            base_ds,
            block_size=data_cfg["spatial_block"]["block_size"],
            n_folds=data_cfg["spatial_block"]["n_folds"],
        )

        n_train_full = len(train_full_ds)
        n_test = len(test_ds)
        if n_train_full == 0:
            raise RuntimeError("Empty training set after spatial blocking.")
        if n_test == 0:
            print("Warning: empty test set after spatial blocking.")

        # Train/val split (on stable keys)
        idx = np.arange(n_train_full)
        train_idx, val_idx = train_test_split(
            idx, test_size=self.val_fraction, random_state=seed
        )

        ref_df = train_full_ds.obs_lookup.reset_index(drop=True)
        keys_train = ref_df.loc[train_idx, ["timestamp", "idx_obs"]].drop_duplicates()
        keys_val = ref_df.loc[val_idx, ["timestamp", "idx_obs"]].drop_duplicates()
        keys_test = test_ds.obs_lookup[["timestamp", "idx_obs"]].drop_duplicates()

        # Write split key CSVs
        out_paths = self._split_paths
        os.makedirs(str(Path(out_paths["train"]).parent), exist_ok=True)
        keys_train.to_csv(out_paths["train"], index=False)
        keys_val.to_csv(out_paths["val"], index=False)
        keys_test.to_csv(out_paths["test"], index=False)

        print("Wrote split keys:")
        for k, p in out_paths.items():
            print(f"  - {k}: {p}")

        print("Preparation complete ✅")

    # ---------------------------
    # setup(): fast if prepared
    # ---------------------------

    def setup(self, stage: Optional[str] = None):
        paths = self.cfg["paths"]
        data_cfg = self.cfg["data"]
        seed = int(self.cfg["experiment"]["seed"])

        madis_fp = _expand_env(paths["madis"])
        terrain_dir = _expand_env(paths["terrain_dir"])
        ifs_dir = _expand_env(paths["ifs_dir"])

        topo_fps = sorted(glob.glob(os.path.join(terrain_dir, "*_reprojected_wgs84_cowy_990m.nc")))
        hrrr_fps = sorted(glob.glob(os.path.join(ifs_dir, "*.nc")))
        max_files = data_cfg.get("max_hrrr_files", None)
        if max_files is not None:
            hrrr_fps = hrrr_fps[:int(max_files)]

        # Base dataset (kept light; actual sample bytes are loaded lazily per __getitem__)
        base_ds = CoWyPointDataset(
            madis_fp=madis_fp,
            hrrr_fps=hrrr_fps,
            topo_fps=topo_fps,
            dist_lim=data_cfg["dist_lim"],
        )

        if self._prepared_exists():
            # ---- FAST PATH: load split keys and subset without recomputing ----
            print("Found prepared split keys — skipping spatial blocking and random split.")
            p = self._split_paths
            keys_train = pd.read_csv(p["train"])
            keys_val = pd.read_csv(p["val"])
            keys_test = pd.read_csv(p["test"])

            # Build datasets via key-based inner join
            train_ds = copy.deepcopy(base_ds)
            val_ds = copy.deepcopy(base_ds)
            test_ds = copy.deepcopy(base_ds)

            train_ds.obs_lookup = (
                train_ds.obs_lookup.merge(keys_train, on=["timestamp", "idx_obs"], how="inner")
                .reset_index(drop=True)
            )
            val_ds.obs_lookup = (
                val_ds.obs_lookup.merge(keys_val, on=["timestamp", "idx_obs"], how="inner")
                .reset_index(drop=True)
            )
            test_ds.obs_lookup = (
                test_ds.obs_lookup.merge(keys_test, on=["timestamp", "idx_obs"], how="inner")
                .reset_index(drop=True)
            )

            if len(train_ds) == 0 or len(val_ds) == 0:
                raise RuntimeError(
                    f"Empty split after applying prepared keys: train={len(train_ds)}, val={len(val_ds)}. "
                    "Check prepared CSVs or data paths."
                )
        else:
            # ---- SLOW PATH (original behavior): compute splits on the fly ----
            print("No prepared split keys found — running spatial blocking and random split.")
            train_full_ds, test_ds = spatial_blocking(
                base_ds,
                block_size=data_cfg["spatial_block"]["block_size"],
                n_folds=data_cfg["spatial_block"]["n_folds"],
            )

            n_train_full = len(train_full_ds)
            n_test = len(test_ds)
            if n_train_full == 0:
                raise RuntimeError("Empty training set after spatial blocking.")
            if n_test == 0:
                print("Warning: empty test set after spatial blocking.")

            print(f"Train (pre val split): {n_train_full:,} rows | Test: {n_test:,} rows")

            idx = np.arange(n_train_full)
            train_idx, val_idx = train_test_split(
                idx, test_size=self.val_fraction, random_state=seed
            )

            ref_df = train_full_ds.obs_lookup.reset_index(drop=True)
            keys_train = ref_df.loc[train_idx, ["timestamp", "idx_obs"]].drop_duplicates()
            keys_val = ref_df.loc[val_idx, ["timestamp", "idx_obs"]].drop_duplicates()

            train_ds = copy.deepcopy(train_full_ds)
            val_ds = copy.deepcopy(train_full_ds)

            train_ds.obs_lookup = (
                train_ds.obs_lookup.merge(keys_train, on=["timestamp", "idx_obs"], how="inner")
                .reset_index(drop=True)
            )
            val_ds.obs_lookup = (
                val_ds.obs_lookup.merge(keys_val, on=["timestamp", "idx_obs"], how="inner")
                .reset_index(drop=True)
            )

            print(f"Train: {len(train_ds):,} rows | Val: {len(val_ds):,} rows (after key-based split)")

        # --- Optional class balancing on train split ---
        if data_cfg.get("balance_classes", False):
            thresholds = data_cfg.get("balance_thresholds", [5, 11])
            train_ds = _balance_train_classes(train_ds, thresholds=thresholds, seed=seed)

        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds

        # Informative print for loader config
        print(
            "DataLoader config → "
            f"num_workers={self.dl_num_workers}, pin_memory={self.dl_pin_memory}, "
            f"persistent_workers={bool(self.dl_num_workers > 0 and self.dl_persistent_workers)}, "
            f"prefetch_factor={self.dl_prefetch_factor if self.dl_num_workers > 0 else None}, "
            f"drop_last={self.dl_drop_last}, "
            f"pin_memory_device={self.dl_pin_memory_device}, "
            f"multiprocessing_context={self.dl_mp_context}"
        )

    # ---------------------------
    # DataLoaders
    # ---------------------------

    def _build_loader_kwargs(self, split: str):
        """
        Build DataLoader kwargs from config with PyTorch-safe guards.
        - shuffle is False because dataset handles timestep-aware shuffling.
        - persistent_workers and prefetch_factor require num_workers > 0.
        """
        kwargs = dict(
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.dl_num_workers,
            pin_memory=self.dl_pin_memory,
            drop_last=self.dl_drop_last,
            worker_init_fn=_seed_worker,
            generator=self._dl_generator,
        )

        # Only set persistent_workers if num_workers > 0
        if self.dl_num_workers > 0 and self.dl_persistent_workers:
            kwargs["persistent_workers"] = True

        # Only set prefetch_factor if num_workers > 0 (PyTorch restriction)
        if self.dl_num_workers > 0 and self.dl_prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(self.dl_prefetch_factor)

        # Optional pin_memory_device (supported in newer PyTorch); safe-guarded
        if self.dl_pin_memory and self.dl_pin_memory_device:
            kwargs["pin_memory_device"] = self.dl_pin_memory_device

        # Optional multiprocessing context if provided (use only if needed)
        if self.dl_mp_context:
            kwargs["multiprocessing_context"] = self.dl_mp_context

        return kwargs

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            **self._build_loader_kwargs(split="train"),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            **self._build_loader_kwargs(split="val"),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            **self._build_loader_kwargs(split="test"),
        )