# datamodule.py

import os
import glob
import copy
import random
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightning as L
import torch
from torch.utils.data import DataLoader

from cowy.data.dataset import CoWyPointDataset
from cowy.data.spatial import spatial_blocking


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


def _balance_train_classes(train_ds, thresholds=[5, 11], seed: int = 42):
    """
    Upsample bins to the size of the largest bin.
    Operates by rewriting train_ds.obs_lookup only (datasets/memory-mapped arrays unchanged).
    """
    print("Balancing training classes …")
    df = train_ds.obs_lookup.copy()

    # Pull the observation values corresponding to obs_lookup rows
    obs_all = train_ds.dset_madis["windspeed_10m"].values  # shape: [time, station]
    obs = obs_all[df["idt_madis"].values, df["idx_obs"].values]  # 1D aligned with df rows

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


class CoWyDataModule(L.LightningDataModule):
    """
    Lightning DataModule with fully config-driven DataLoader options suitable for GPU clusters.

    YAML keys under `data:` that this module reads:
      - num_workers: int (default 0)
      - pin_memory: bool (default True)
      - persistent_workers: bool (default False; effective only if num_workers > 0)
      - prefetch_factor: int|None (effective only if num_workers > 0)
      - drop_last: bool (default False)
      - pin_memory_device: str|None (e.g., "cuda", only for recent PyTorch versions)
      - multiprocessing_context: str|None ("fork", "spawn", "forkserver") — optional

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

    def _build_loader_kwargs(self, split: str):
        """
        Build DataLoader kwargs from config with PyTorch-safe guards.
        - shuffle is False because dataset handles timestep-aware shuffling.
        - persistent_workers and prefetch_factor require num_workers > 0.
        """
        kwargs = dict(
            batch_size=self.batch_size,
            shuffle=False,  # dataset handles internal shuffling
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

        # Split-specific hooks could go here, if you ever need them
        # if split == "train": ...
        return kwargs

    def setup(self, stage: Optional[str] = None):
        paths = self.cfg["paths"]
        data_cfg = self.cfg["data"]
        seed = int(self.cfg["experiment"]["seed"])

        madis_fp = _expand_env(paths["madis"])
        terrain_dir = _expand_env(paths["terrain_dir"])
        ifs_dir = _expand_env(paths["ifs_dir"])

        topo_fps = sorted(glob.glob(os.path.join(terrain_dir, "*_reprojected_wgs84_cowy_990m.nc")))

        # Allow limiting the number of files via config (optional)
        hrrr_fps = sorted(glob.glob(os.path.join(ifs_dir, "*.nc")))
        max_files = data_cfg.get("max_hrrr_files", None)
        if max_files is not None:
            hrrr_fps = hrrr_fps[:int(max_files)]

        # --- Build the base dataset ---
        base_ds = CoWyPointDataset(
            madis_fp=madis_fp,
            hrrr_fps=hrrr_fps,
            topo_fps=topo_fps,
            dist_lim=data_cfg["dist_lim"],
        )

        # --- Spatial blocking to get train/test ---
        train_full_ds, test_ds = spatial_blocking(
            base_ds,
            block_size=data_cfg["spatial_block"]["block_size"],
            n_folds=data_cfg["spatial_block"]["n_folds"],
        )

        # Sanity
        n_train_full = len(train_full_ds)
        n_test = len(test_ds)
        if n_train_full == 0:
            raise RuntimeError("Empty training set after spatial blocking.")
        if n_test == 0:
            print("Warning: empty test set after spatial blocking.")

        print(f"Train (pre val split): {n_train_full:,} rows | Test: {n_test:,} rows")

        # --- Split train_full into train/val using stable keys, not positions ---
        idx = np.arange(n_train_full)
        train_idx, val_idx = train_test_split(
            idx, test_size=self.val_fraction, random_state=seed
        )

        ref_df = train_full_ds.obs_lookup.reset_index(drop=True)
        # Keys that uniquely tie an observation to MADIS+HRRR cell at a given time
        keys_train = ref_df.loc[train_idx, ["timestamp", "idx_obs"]].drop_duplicates()
        keys_val = ref_df.loc[val_idx, ["timestamp", "idx_obs"]].drop_duplicates()

        # Build two independent copies before subsetting
        train_ds = copy.deepcopy(train_full_ds)
        val_ds = copy.deepcopy(train_full_ds)

        # Subset via key-based inner joins (robust to reordering)
        train_ds.obs_lookup = (
            train_ds.obs_lookup.merge(keys_train, on=["timestamp", "idx_obs"], how="inner")
            .reset_index(drop=True)
        )
        val_ds.obs_lookup = (
            val_ds.obs_lookup.merge(keys_val, on=["timestamp", "idx_obs"], how="inner")
            .reset_index(drop=True)
        )

        if len(train_ds) == 0 or len(val_ds) == 0:
            raise RuntimeError(
                f"Empty split detected: train={len(train_ds)}, val={len(val_ds)}. "
                f"Check spatial blocking and split fraction."
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