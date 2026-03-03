
#!/usr/bin/env python3
"""
Evaluate a trained checkpoint:
 - Finds the latest/best checkpoint automatically
 - Runs forward passes on train/val/test
 - Saves predictions, obs, and obs_lookup splits
 - Computes binned metrics and writes JSON + text table
 - Optionally makes a map plot of train/test coverage
"""

import argparse
import glob
import json
import os
import re
import matplotlib.pyplot as plt
from datetime import datetime

import numpy as np
import yaml
import torch
import lightning as L

from cowy.training.datamodule import CoWyDataModule
from cowy.models.point_model import PointCorrectionModel
from cowy.evaluation.metrics import evaluate_bins, format_results_table
from cowy.evaluation.plots import plot_train_test_map, density_scatter, confusion_matrix, plot_confusion_matrix


def _expand_env(s: str) -> str:
    return os.path.expandvars(os.path.expanduser(s))


def find_best_checkpoint(output_root: str, exp_name: str) -> tuple[str, str]:
    """
    Returns (ckpt_path, version_dir)
    """
    exp_dir = os.path.join(output_root, exp_name)
    if not os.path.isdir(exp_dir):
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    versions = sorted(glob.glob(os.path.join(exp_dir, "version_*")))
    if not versions:
        raise FileNotFoundError(f"No version directories found in: {exp_dir}")

    version_dir = versions[-1]  # most recent by lexicographic order
    ckpts = sorted(glob.glob(os.path.join(version_dir, "checkpoints", "*.ckpt")))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in: {os.path.join(version_dir, 'checkpoints')}")

    # Pick last modified as "best" (simple heuristic)
    ckpt_path = max(ckpts, key=os.path.getmtime)
    return ckpt_path, version_dir


def forward_pass(model: L.LightningModule, dataloader: torch.utils.data.DataLoader):
    preds, obs, inputs = [], [], []
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y_hat = model(x).cpu().numpy()
            preds.append(y_hat)
            obs.append(y.numpy())
            inputs.append(x.cpu().numpy())
    return np.vstack(preds), np.vstack(obs), np.vstack(inputs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config")
    parser.add_argument("--ckpt", default=None, help="Path to .ckpt (optional; auto-discovery if omitted)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_root = _expand_env(cfg["paths"]["output_root"])
    exp_name = cfg["experiment"]["name"]

    if args.ckpt:
        ckpt_path = _expand_env(args.ckpt)
        version_dir = os.path.dirname(os.path.dirname(ckpt_path))  # …/version_X/checkpoints/best.ckpt
    else:
        ckpt_path, version_dir = find_best_checkpoint(out_root, exp_name)

    print(f"Using checkpoint: {ckpt_path}")
    print(f"Version dir:      {version_dir}")

    # Data + model
    dm = CoWyDataModule(cfg)
    dm.setup("fit")  # build train/val/test

    mean = np.load(_expand_env(cfg["paths"]["mean_file"]))
    std  = np.load(_expand_env(cfg["paths"]["std_file"]))

    model = PointCorrectionModel.load_from_checkpoint(
        ckpt_path,
        cfg=cfg,
        mean=mean,
        std=std,
        map_location="cpu",
    )

    trainer = L.Trainer(logger=False, enable_checkpointing=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Forward passes
    print("Forward pass: train/val/test …")
    pred_train, obs_train, inputs_train = forward_pass(model, dm.train_dataloader())
    pred_val,   obs_val,   inputs_val   = forward_pass(model, dm.val_dataloader())
    pred_test,  obs_test,  inputs_test  = forward_pass(model, dm.test_dataloader())

    # Save arrays
    results_dir = os.path.join(version_dir, "best_ckpt_results")
    os.makedirs(results_dir, exist_ok=True)

    np.save(os.path.join(results_dir, "pred_train.npy"), pred_train)
    np.save(os.path.join(results_dir, "obs_train.npy"),  obs_train)
    np.save(os.path.join(results_dir, "inputs_train.npy"), inputs_train)
    np.save(os.path.join(results_dir, "pred_val.npy"),   pred_val)
    np.save(os.path.join(results_dir, "obs_val.npy"),    obs_val)
    np.save(os.path.join(results_dir, "inputs_val.npy"),  inputs_val)
    np.save(os.path.join(results_dir, "pred_test.npy"),  pred_test)
    np.save(os.path.join(results_dir, "obs_test.npy"),   obs_test)
    np.save(os.path.join(results_dir, "inputs_test.npy"), inputs_test)

    # Save baseline forecast (ws_10) extracted from inputs_*
    try:
        hrrr_vars = list(dm.train_ds.vars_hrrr)
        if "ws_10m" not in hrrr_vars:
            raise KeyError(f"'ws_10m' not found in dataset vars_hrrr. Available: {hrrr_vars}")

        ws_idx = hrrr_vars.index("ws_10m")
        print(f"Baseline forecast feature: ws_10m at inputs column index {ws_idx}")

        # (N,1) arrays to match pred/obs shapes
        fcst_train = inputs_train[:, ws_idx:ws_idx + 1]
        fcst_val   = inputs_val[:,   ws_idx:ws_idx + 1]
        fcst_test  = inputs_test[:,  ws_idx:ws_idx + 1]

        np.save(os.path.join(results_dir, "ws_forecast_train.npy"), fcst_train)
        np.save(os.path.join(results_dir, "ws_forecast_val.npy"),   fcst_val)
        np.save(os.path.join(results_dir, "ws_forecast_test.npy"),  fcst_test)

        # Save metadata so you can reconstruct feature meaning later
        with open(os.path.join(results_dir, "feature_index.json"), "w") as f:
            json.dump({"ws_10m_index": ws_idx, "hrrr_vars": hrrr_vars}, f, indent=2)

    except Exception as e:
        print(f"Saving baseline forecast (ws_10m) failed (skipping): {e}")

    # Save obs_lookup splits
    dm.train_ds.obs_lookup.to_csv(os.path.join(results_dir, "train_obs_lookup.csv"), index=False)
    dm.val_ds.obs_lookup.to_csv(os.path.join(results_dir, "val_obs_lookup.csv"), index=False)
    dm.test_ds.obs_lookup.to_csv(os.path.join(results_dir, "test_obs_lookup.csv"), index=False)

    # Metrics
    thresholds = cfg["evaluation"].get("bins", [5, 10])
    train_stats = evaluate_bins(pred_train, obs_train, thresholds)
    val_stats   = evaluate_bins(pred_val,   obs_val,   thresholds)
    test_stats  = evaluate_bins(pred_test,  obs_test,  thresholds)

    metrics = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "bins": thresholds,
        "train": train_stats,
        "val": val_stats,
        "test": test_stats,
    }
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    table = []
    table.append("=== Train ===\n" + format_results_table(train_stats))
    table.append("\n=== Val ===\n" + format_results_table(val_stats))
    table.append("\n=== Test ===\n" + format_results_table(test_stats))
    joined = "\n".join(table)
    with open(os.path.join(results_dir, "metrics.txt"), "w") as f:
        f.write(joined)
    print(joined)

    # Plot map (optional)
    plot_dir = os.path.join(results_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    if cfg.get("evaluation", {}).get("plots", {}).get("map", {}).get("enabled", True):
        png = os.path.join(plot_dir, "train_test_map.png")
        try:
            ax = plot_train_test_map(dm.train_ds, dm.test_ds, dem_path=cfg["paths"].get("dem"))
            ax.set_title("Training and Test Set Observations")
            plt.tight_layout()
            plt.savefig(png, dpi=200)
            plt.close(ax.figure)
            print(f"Saved map: {png}")
        except Exception as e:
            print(f"Map plotting failed (skipping): {e}")

    if cfg.get("evaluation", {}).get("plots", {}).get("scatter", {}).get("enabled", True):
            try:
                error_ml = pred_test - obs_test
                error_bl = fcst_test - obs_test

                year = re.search(r'\d{4}', cfg["paths"]["madis"]).group()

                # baseline scatter
                fig, ax = plt.subplots(figsize=(7, 6))
                ax = density_scatter(x=obs_test, y=fcst_test)
                ax.set_xlabel("Observed 10m Windspeed (m/s)")
                ax.set_ylabel("IFS 10m Windspeed (m/s)")
                ax.set_title(f"IFS f72-96 {year}")
                plt.tight_layout()
                png = os.path.join(plot_dir, "scatter_bl_test.png")
                plt.savefig(png, dpi=150, bbox_inches="tight")
                print(f"BL Saved scatter: {png}")
                plt.close()

                # ML scatter 
                fig, ax = plt.subplots(figsize=(7, 6))
                ax = density_scatter(x=obs_test, y=pred_test)
                ax.set_xlabel("Observed 10m Windspeed (m/s)")
                ax.set_ylabel("ML predicted 10m Windspeed (m/s)")
                ax.set_title(f"IFS f72-96 {year}")
                plt.tight_layout()
                png = os.path.join(plot_dir, "scatter_ml_test.png")
                plt.savefig(png, dpi=150, bbox_inches="tight")
                print(f"Saved ML scatter: {png}")
                plt.close()

                # Baseline error scatter
                fig, ax = plt.subplots(figsize=(7, 6))
                ax = density_scatter(x=obs_test, y=error_bl, one_to_one_line=False, trend_line=True, cmap="plasma")
                ax.set_xlabel("Observed 10m Windspeed (m/s)")
                ax.set_ylabel("Error (m/s)")
                ax.set_title(f"IFS f72-96 {year} Error")
                plt.tight_layout()
                png = os.path.join(plot_dir, "scatter_error_bl_test.png")
                plt.savefig(png, dpi=150, bbox_inches="tight")
                print(f"Saved BL scatter error: {png}")
                plt.close()

                #ML error scatter
                fig, ax = plt.subplots(figsize=(7, 6))
                ax = density_scatter(x=obs_test, y=error_ml, one_to_one_line=False, trend_line=True, cmap="plasma")
                ax.set_xlabel("Observed 10m Windspeed (m/s)")
                ax.set_ylabel("Error (m/s)")
                ax.set_title(f"IFS f72-96 {year} Error")
                plt.tight_layout()
                png = os.path.join(plot_dir, "scatter_error_ml_test.png")
                plt.savefig(png, dpi=150, bbox_inches="tight")
                print(f"Saved ML scatter error: {png}")
                plt.close()

            except Exception as e:
                print(f"Scatter plotting failed (skipping): {e}")

    if cfg.get("evaluation", {}).get("plots", {}).get("confusion_matrix", {}).get("enabled", True):
            try:
                threshold = cfg["evaluation"].get("threshold", 15.0)

                # ML confusion matrix
                cm_df = confusion_matrix(obs_test, pred_test, threshold=threshold)
                plot_confusion_matrix(cm_df, title=f"ML Confusion Matrix — threshold {threshold} m/s")
                plt.tight_layout()
                png = os.path.join(plot_dir, "confusion_matrix_ml_test.png")
                plt.savefig(png, dpi=150, bbox_inches="tight")
                print(f"Saved ML confusion matrix: {png}")
                plt.close()

                # Baseline confusion matrix
                cm_df_bl = confusion_matrix(obs_test, fcst_test, threshold=threshold)
                plot_confusion_matrix(cm_df_bl, title=f"Baseline Confusion Matrix — threshold {threshold} m/s")
                plt.tight_layout()
                png = os.path.join(plot_dir, "confusion_matrix_bl_test.png")
                plt.savefig(png, dpi=150, bbox_inches="tight")
                print(f"Saved baseline confusion matrix: {png}")
                plt.close()

            except Exception as e:
                print(f"Confusion matrix plotting failed (skipping): {e}")

if __name__ == "__main__":
    main()
