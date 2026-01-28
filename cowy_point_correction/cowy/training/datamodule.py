
import os
import glob
import copy
from typing import Optional

import numpy as np
import yaml
from sklearn.model_selection import train_test_split
import lightning as L
from torch.utils.data import DataLoader

from cowy.data.dataset import CoWyPointDataset
from cowy.data.spatial import spatial_blocking


def _expand_env(s: str) -> str:
    return os.path.expandvars(os.path.expanduser(s))


def _balance_train_classes(train_ds, thresholds=[5, 11]):
    """
    Upsample bins to the size of the largest bin.
    train_ds.obs_lookup is expanded; train_ds internals unchanged.
    """
    print("Balancing training classes …")
    df = train_ds.obs_lookup.copy()

    obs_all = train_ds.dset_madis["windspeed_10m"].values
    obs = obs_all[df["idt_madis"].values, df["idx_obs"].values]

    thresholds = sorted(thresholds)
    edges = [0] + thresholds + [np.inf]

    bins = []
    sizes = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (obs >= lo) & (obs < hi)
        bins.append(df[mask])
        sizes.append(mask.sum())

    max_size = max(sizes) if sizes else 0
    print(f"Target bin size: {max_size}")

    balanced = []
    for (lo, hi), part, size in zip(zip(edges[:-1], edges[1:]), bins, sizes):
        if size == 0:
            continue
        if size < max_size:
            up = part.sample(n=max_size, replace=True, random_state=42)
            balanced.append(up)
        else:
            balanced.append(part)

    train_ds.obs_lookup = ( 
        copy.deepcopy(np.random.RandomState(42))  # keep deterministic
    ) and ( 
        # combine and shuffle
        __import__("pandas").concat(balanced, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    )
    print(f"Balanced training size: {len(train_ds)}")
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

        madis_fp = _expand_env(paths["madis"])
        terrain_dir = _expand_env(paths["terrain_dir"])
        ifs_dir = _expand_env(paths["ifs_dir"])

        topo_fps = sorted(glob.glob(os.path.join(terrain_dir, "*_reprojected_wgs84_cowy_990m.nc")))
        hrrr_fps = sorted(glob.glob(os.path.join(ifs_dir, "*.nc")))[:20]

        base_ds = CoWyPointDataset(
            madis_fp=madis_fp,
            hrrr_fps=hrrr_fps,
            topo_fps=topo_fps,
            dist_lim=data_cfg["dist_lim"],
        )

        # Spatial blocking to create test set
        train_ds, test_ds = spatial_blocking(
            base_ds,
            block_size=data_cfg["spatial_block"]["block_size"],
            n_folds=data_cfg["spatial_block"]["n_folds"],
        )

        # Split train into train/val by index
        idx = np.arange(len(train_ds))
        train_idx, val_idx = train_test_split(idx, test_size=self.val_fraction, random_state=self.cfg["experiment"]["seed"])

        train_ds.obs_lookup = train_ds.obs_lookup.iloc[train_idx].reset_index(drop=True)
        val_ds = copy.deepcopy(train_ds)
        val_ds.obs_lookup = val_ds.obs_lookup.iloc[val_idx].reset_index(drop=True)

        # Optional class balancing (upsample)
        if self.cfg["data"].get("balance_classes", False):
            thresholds = self.cfg["data"].get("balance_thresholds", [5, 11])
            train_ds = _balance_train_classes(train_ds, thresholds)

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
