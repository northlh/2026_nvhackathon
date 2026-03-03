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

    # def _loss(self, yhat, y):
    #     return nn.functional.mse_loss(yhat, y)
    
    def _loss(self, yhat, y, eps: float = 1e-6):
        """Log(ws)-weighted MSE (weights derived from target windspeed)."""
        yhat = yhat.float()
        y = y.float()
        se = (y - yhat) ** 2
        w = 1.0 + torch.log1p(torch.clamp(y, min=0.0))
        w = w / (w.mean().clamp_min(eps))
        return (w * se).mean()
    
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

    # def validation_step(self, batch, _):
    #     x, y = batch
    #     loss = self._loss(self(x), y)
    #     self.log("validation_loss", loss, prog_bar=True)
        
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
        return loss
    
    def configure_optimizers(self):
        opt = AdamW(
            self.parameters(),
            lr=self.hparams.cfg["training"]["optimizer"]["learning_rate"],
            eps=self.hparams.cfg["training"]["optimizer"]["eps"],
            weight_decay=self.hparams.cfg["training"]["optimizer"]["weight_decay"],
        )
        return opt