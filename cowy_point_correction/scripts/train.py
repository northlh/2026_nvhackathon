# Ensure repo root is on sys.path regardless of CWD
import os
import sys
import argparse
import json
import yaml
import warnings
import torch
torch.set_float32_matmul_precision('high')
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks import RichProgressBar

# Silence xarray timedelta warning
warnings.filterwarnings(
    "ignore",
    message=".*decode the variable 'step' into a timedelta64 dtype.*"
)

# Add repository root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from cowy.training.datamodule import CoWyDataModule
from cowy.training.callbacks import build_callbacks
from cowy.models.point_model import PointCorrectionModel


# -----------------------
# Utilities
# -----------------------

def expand(path: str) -> str:
    return os.path.expandvars(os.path.expanduser(path))


def parse_args():
    parser = argparse.ArgumentParser(description="Train or prepare CoWy model")
    parser.add_argument("config", type=str, help="Path to YAML config")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Run data preparation only, then exit",
    )
    return parser.parse_args()


# -----------------------
# Main training function
# -----------------------

def train(cfg):
    # ---- Data module (FAST path; assumes prepare() already ran) ----
    print("Initializing datamodule...")
    dm = CoWyDataModule(cfg)
    dm.setup("fit")

    # ---- Load normalization statistics ----
    import numpy as np
    mean = np.load(expand(cfg["paths"]["mean_file"]), mmap_mode="r")
    std = np.load(expand(cfg["paths"]["std_file"]), mmap_mode="r")

    # ---- Model ----
    print("Building model...")
    model = PointCorrectionModel(cfg, mean, std)

    # ---- Logging ----
    output_root = expand(cfg["paths"]["output_root"])
    exp_name = cfg["experiment"]["name"]

    tb_logger = TensorBoardLogger(output_root, name=exp_name)
    csv_logger = CSVLogger(output_root, name=exp_name, version=tb_logger.version)

    # ---- Callbacks ----
    callbacks = build_callbacks(cfg)
    callbacks.append(
        RichProgressBar(
            refresh_rate=cfg.get("logging", {}).get("progress_refresh_rate", 1)
        )
    )

    # ---- Trainer ----
    trainer = L.Trainer(
        max_epochs=cfg["training"]["max_epochs"],
        accelerator="gpu",
        logger=[tb_logger, csv_logger],
        callbacks=callbacks,
        log_every_n_steps=cfg["logging"]["log_every_n_steps"],
        enable_progress_bar=True,
        precision=cfg["training"]["precision"], ### delete if broken
        num_sanity_val_steps=0,  # avoids slow startup
    )

    # ---- Train ----
    print("Starting training...")
    trainer.fit(model, datamodule=dm)

    # ---- Save config with run ----
    run_dir = os.path.join(
        output_root,
        exp_name,
        f"version_{tb_logger.version}",
    )
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"Training complete. Artifacts saved to {run_dir}")


# -----------------------
# Entrypoint
# -----------------------

if __name__ == "__main__":
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ---- One-time data preparation ----
    if args.prepare_only:
        print("Running DATA PREPARATION ONLY...")
        dm = CoWyDataModule(cfg)
        dm.prepare()  # <--- you implement this
        print("✅ Data preparation complete.")
        sys.exit(0)

    # ---- Normal training ----
    train(cfg)