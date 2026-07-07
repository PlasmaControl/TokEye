"""step_6 — leakage-free out-of-fold (OOF) probabilities for label refinement.

K-fold cross-validation where each fold's model predicts ONLY its held-out
samples, so every sample gets exactly one probability from a model that never
saw it. This replaces the old step_6b/6c refiner, which (a) predicted the
full dataset with every fold and averaged — 4/5 of each "refined" label came
from models that had memorized that sample — and (b) added pixels where the
MC-dropout ensemble was most *uncertain* (entropy > threshold peaks at
p = 0.5). No MC-dropout here: one inference pass per sample.

Outputs (``step_6.h5``):
- ``p_oof``          (N, 2, F, T) float32 — out-of-fold sigmoid probabilities
- ``disagreement``   (N, 2, F, T) float32 — |p_oof - y0|, the intern's visual
  QC map (replaces the deleted manual mask editor)
- ``fold``           (N,) int8 — which fold held each sample out
- attrs: per-fold validation dice/iou

The soft target q = (1 - model_trust)*y0 + model_trust*p_oof is deliberately
NOT stored here: model_trust is an intern knob, and applying it downstream
(step_7) means tuning it re-trains one final model instead of five folds.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import h5py
import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, Subset

from tokeye.models.modules.unet import UNet

from .utils.augmentations import get_augmentation
from .utils.losses import dice_coefficient, get_loss_function, iou_score

if TYPE_CHECKING:
    from pathlib import Path

# Legacy API on purpose: Lightning still calls get_float32_matmul_precision(),
# and mixing it with the new torch.backends.*.fp32_precision API raises.
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True

logger = logging.getLogger(__name__)

# Augmentation + loss knobs the legacy refiner trained with (kept fixed —
# they are not intern knobs).
_TRAIN_DEFAULTS = {
    "augmentation": True,
    "aug_rotation_degrees": 180,
    "aug_prob_flip": 0.5,
    "aug_elastic": True,
    "aug_elastic_alpha": 50.0,
    "aug_elastic_sigma": 5.0,
    "aug_scale_range": [0.8, 1.2],
    "aug_intensity": True,
    "aug_brightness_range": [0.8, 1.2],
    "aug_contrast_range": [0.8, 1.2],
    "aug_noise_std": 0.05,
    "aug_blur_prob": 0.3,
    "aug_blur_sigma_range": [0.5, 1.5],
    "aug_gamma_range": [0.8, 1.2],
    "aug_apply_prob": 0.8,
    "specaugment": True,
    "specaug_time_warp_W": 20,
    "specaug_freq_mask_F": 5,
    "specaug_time_mask_T": 5,
    "specaug_freq_mask_num": 0,
    "specaug_time_mask_num": 0,
    "label_smoothing": 0.1,
    "symmetric_alpha": 0.1,
    "symmetric_beta": 1.0,
    "symmetric_weight": 0.5,
    "dice_weight": 0.5,
    "bce_weight": 0.5,
    "focal_alpha": 0.25,
    "focal_gamma": 2.0,
    "focal_weight": 0.5,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "lr_factor": 0.5,
    "lr_patience": 5,
    "lr_min": 1e-6,
}


class DatasetH5(Dataset):
    """(image, mask) pairs from step_5.h5, lazy per-worker file handle."""

    def __init__(self, h5_path: Path, transform=None) -> None:
        self.h5_path = h5_path
        self.transform = transform
        with h5py.File(h5_path, "r") as f:
            self.n = int(f.attrs["n_samples"])
        self._file: h5py.File | None = None

    def __len__(self) -> int:
        return self.n

    def _f(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r", swmr=True)
        return self._file

    def __getitem__(self, idx: int):
        f = self._f()
        x = torch.from_numpy(np.asarray(f["images"][str(idx)])).float()
        y = torch.from_numpy(np.asarray(f["masks"][str(idx)])).float()
        if self.transform is not None:
            x, y = self.transform(x, y)
        return x, y


class RefineModule(L.LightningModule):
    def __init__(self, settings: dict) -> None:
        super().__init__()
        self.unet = UNet(
            in_channels=1,
            out_channels=2,
            num_layers=settings["num_layers"],
            first_layer_size=settings["first_layer_size"],
        )
        if not settings["smoke"] and torch.cuda.is_available():
            self.unet.compile()
        loss_settings = {**_TRAIN_DEFAULTS, "loss_type": settings["loss_type"]}
        self.loss_fn = get_loss_function(loss_settings)
        self.lr_settings = loss_settings

    def forward(self, x):
        return self.unet(x)

    def training_step(self, batch, _):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        probs = torch.sigmoid(y_hat)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_dice", dice_coefficient(probs, y))
        self.log("val_iou", iou_score(probs, y))
        return loss

    def predict_step(self, batch, _):
        x, _y = batch
        return torch.sigmoid(self(x))

    def configure_optimizers(self):
        s = self.lr_settings
        opt = torch.optim.Adam(
            self.parameters(), lr=s["learning_rate"], weight_decay=s["weight_decay"]
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=s["lr_factor"], patience=s["lr_patience"], min_lr=s["lr_min"]
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"},
        }


def _loader(dataset, settings: dict, shuffle: bool) -> DataLoader:
    workers = settings["num_workers"]
    return DataLoader(
        dataset,
        batch_size=settings["batch_size"],
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
    )


def main(settings: dict) -> None:
    L.seed_everything(42)
    in_h5 = settings["in_h5"]
    ckpt_dir = settings["ckpt_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_full = DatasetH5(in_h5, transform=get_augmentation(_TRAIN_DEFAULTS))
    eval_full = DatasetH5(in_h5, transform=None)
    n = len(eval_full)
    sample_shape = eval_full[0][1].shape  # (2, F, T)

    with h5py.File(settings["out_h5"], "w") as out:
        p_oof = out.create_dataset(
            "p_oof",
            shape=(n, *sample_shape),
            dtype=np.float32,
            chunks=(1, *sample_shape),
            compression="lzf",
        )
        disagreement = out.create_dataset(
            "disagreement",
            shape=(n, *sample_shape),
            dtype=np.float32,
            chunks=(1, *sample_shape),
            compression="lzf",
        )
        fold_of = out.create_dataset("fold", shape=(n,), dtype=np.int8)

        kfold = KFold(n_splits=settings["n_folds"], shuffle=True, random_state=42)
        fold_metrics = []
        for fold, (train_idx, val_idx) in enumerate(kfold.split(np.arange(n))):
            logger.info(
                f"fold {fold}: train={len(train_idx)}, held-out={len(val_idx)}"
            )
            module = RefineModule(settings)
            ckpt = ModelCheckpoint(
                dirpath=ckpt_dir / f"fold_{fold}",
                monitor="val_loss",
                save_top_k=1,
            )
            trainer = L.Trainer(
                max_epochs=settings["max_epochs"],
                precision=settings["precision"],
                accelerator="auto",
                devices=1,
                callbacks=[
                    ckpt,
                    EarlyStopping(monitor="val_loss", patience=3),
                ],
                enable_progress_bar=False,
                log_every_n_steps=10,
                logger=False,
            )
            trainer.fit(
                module,
                train_dataloaders=_loader(
                    Subset(train_full, train_idx), settings, shuffle=True
                ),
                val_dataloaders=_loader(
                    Subset(eval_full, val_idx), settings, shuffle=False
                ),
            )
            fold_metrics.append(
                {
                    "fold": fold,
                    "val_loss": float(trainer.callback_metrics["val_loss"]),
                    "val_dice": float(trainer.callback_metrics["val_dice"]),
                    "val_iou": float(trainer.callback_metrics["val_iou"]),
                }
            )

            # Predict ONLY the held-out fold — this is the leakage fix.
            preds = trainer.predict(
                module,
                dataloaders=_loader(
                    Subset(eval_full, val_idx), settings, shuffle=False
                ),
                ckpt_path=ckpt.best_model_path or None,
            )
            probs = torch.cat(preds).float().cpu().numpy()
            for j, global_idx in enumerate(val_idx):
                y0 = np.asarray(eval_full._f()["masks"][str(int(global_idx))])
                p_oof[global_idx] = probs[j]
                disagreement[global_idx] = np.abs(probs[j] - y0)
                fold_of[global_idx] = fold
            logger.info(f"fold {fold} OOF written ({len(val_idx)} samples)")

        out.attrs["n_folds"] = settings["n_folds"]
        out.attrs["fold_metrics"] = json.dumps(fold_metrics)

    # Every sample must have been predicted by exactly one fold.
    counts = np.zeros(n, dtype=int)
    for _, val_idx in KFold(
        n_splits=settings["n_folds"], shuffle=True, random_state=42
    ).split(np.arange(n)):
        counts[val_idx] += 1
    if not np.all(counts == 1):
        raise RuntimeError("OOF coverage broken: some samples not predicted once")
    logger.info(f"step_6 complete: OOF coverage verified for {n} samples")
