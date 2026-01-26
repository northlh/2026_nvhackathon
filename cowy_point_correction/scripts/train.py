
# scripts/train.py
import os
import json
import yaml
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

from cowy.training.datamodule import CoWyDataModule
from cowy.training.callbacks import build_callbacks
from cowy.models.point_model import PointCorrectionModel


def expand(s):
    return os.path.expandvars(os.path.expanduser(s))


def main(cfg):
    dm = CoWyDataModule(cfg)
    dm.setup("fit")

    mean = __import__("numpy").load(expand(cfg["paths"]["mean_file"]))
    std = __import__("numpy").load(expand(cfg["paths"]["std_file"]))

    model = PointCorrectionModel(cfg, mean, std)

    root = expand(cfg["paths"]["output_root"])
    name = cfg["experiment"]["name"]

    tb = TensorBoardLogger(root, name=name)
    csv = CSVLogger(root, name=name, version=tb.version)

    callbacks = build_callbacks(cfg)

    trainer = L.Trainer(
        max_epochs=cfg["training"]["max_epochs"],
        accelerator="auto",
        logger=[tb, csv],
        callbacks=callbacks,
        log_every_n_steps=cfg["logging"]["log_every_n_steps"],
    )

    trainer.fit(model, dm)

    # save config with run
    run_dir = os.path.join(root, name, f"version_{tb.version}")
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


if __name__ == "__main__":
    with open(os.sys.argv[1]) as f:
        cfg = yaml.safe_load(f)
    main(cfg)
