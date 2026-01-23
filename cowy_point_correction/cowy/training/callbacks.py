
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

def build_callbacks(cfg: dict, version_dir: str | None = None):
    cbs = []
    es_cfg = cfg["training"].get("early_stopping", {})
    if es_cfg.get("enabled", True):
        cbs.append(EarlyStopping(
            monitor=es_cfg.get("monitor", "validation_loss"),
            min_delta=es_cfg.get("min_delta", 0.0),
            patience=es_cfg.get("patience", 50),
            mode="min",
            verbose=False,
        ))

    ck_cfg = cfg.get("logging", {}).get("checkpoint", {})
    cbs.append(ModelCheckpoint(
        monitor=ck_cfg.get("monitor", "validation_loss"),
        mode=ck_cfg.get("mode", "min"),
        save_top_k=ck_cfg.get("save_top_k", 1),
    ))
    return cbs