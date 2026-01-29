import os
import glob
import copy
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightning as L
from torch.utils.data import DataLoader

from cowy.data.dataset import CoWyPointDataset
from cowy.data.spatial import spatial_blocking


def _expand_env(s: str) -> str:
    return os.path.expandvars(os.path.expanduser(s))


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
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

        self.batch_size = cfg["training"]["batch_size"]
        self.num_workers = cfg["data"].get("num_workers", 0)
        self.pin_memory = cfg["data"].get("pin_memory", True)
        self.val_fraction = cfg["data"]["val_fraction"]

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

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
        # We still choose random positions, but translate them to keys and merge.
        idx = np.arange(n_train_full)
        train_idx, val_idx = train_test_split(
            idx, test_size=self.val_fraction, random_state=seed
        )

        ref_df = train_full_ds.obs_lookup.reset_index(drop=True)
        # Keys that uniquely tie an observation to MADIS+HRRR cell at a given time
        # (timestamp + idx_obs are sufficient given how obs_lookup is built)
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

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=False,  # shuffle handled inside dataset (timestep-aware)
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )