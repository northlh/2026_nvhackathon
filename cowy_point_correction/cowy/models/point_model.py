# cowy/models/point_model.py
import torch
import torch.nn as nn
import lightning as L
from torch.optim import AdamW

from cowy.models.normalization import FixedNorm


class PointCorrectionModel(L.LightningModule):

    def __init__(self, cfg, mean, std):
        super().__init__()
        self.save_hyperparameters(ignore=["mean", "std"])

        nf = cfg["model"]["n_filters"]
        p = cfg["model"]["dropout"]

        self.model = nn.Sequential(
            FixedNorm(mean, std),
            *[
                layer
                for _ in range(4)
                for layer in (
                    nn.Linear(nf if _ else len(mean), nf),
                    nn.LeakyReLU(),
                    nn.Dropout(p),
                )
            ],
            nn.Linear(nf, 1),
        )

    def forward(self, x):
        return self.model(x)

    def _loss(self, yhat, y):
        return nn.functional.mse_loss(yhat, y)

    def training_step(self, batch, _):
        x, y = batch
        loss = self._loss(self(x), y)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        loss = self._loss(self(x), y)
        self.log("validation_loss", loss, prog_bar=True)
    def validation_step(self, batch, _):
        x, y = batch
        loss = self._loss(self(x), y)
        self.log(
            "validation_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
    def configure_optimizers(self):
        opt = AdamW(
            self.parameters(),
            lr=self.hparams.cfg["training"]["optimizer"]["learning_rate"],
            eps=self.hparams.cfg["training"]["optimizer"]["eps"],
            weight_decay=self.hparams.cfg["training"]["optimizer"]["weight_decay"],
        )
        return opt