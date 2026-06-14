"""Step 3a: Correlation analysis via cross-channel denoising UNet.

Key changes from original ``big_tf_unet`` version:
- ``first_layer_size`` upgraded from 16 to **32** (matches refiner/final).
- ``clamp_range`` computed automatically from data quantiles.
- Bin masking indices detected automatically; mask value = data mean.
- ``total_channels`` and ``adjacent_channels`` derived from modality config.
- Reads from / writes to HDF5 step files instead of individual joblibfiles.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import h5py
import lightning as L
import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image import TotalVariation

from tokeye.models.modules.unet import UNet

from .utils.auto_params import compute_clamp_range, detect_edge_bins
from .utils.configuration import load_settings
from .utils.hdf5_io import create_step_file, write_sample

# TF32: fp32 storage + tensor-core matmul/conv (10-bit operands, fp32 accumulate)
# -- ~bf16 speed, ~fp32 stability. MUST use the legacy API: Lightning calls
# torch.get_float32_matmul_precision() internally, so the new backends.*.fp32_precision
# API raises "mix of legacy and new APIs" at Trainer setup.
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True

L.seed_everything(42)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

default_settings = {
    "adjacent_channels": 3,
    "total_channels": 8,
    "clamp_range": "auto",
    "clamp_percentiles": [1, 99],
    "bin_masking": "auto",
    "bin_mask_value": "mean",
    "gradient_threshold": 0.5,
    "num_layers": 5,
    "first_layer_size": 32,  # upgraded from 16
    "batch_size": 36,
    "num_workers": 8,
    "prefetch_factor": 2,
    "tv_early_stopping": True,
    "tv_patience": 3,
    "max_epochs": 30,
    "precision": "bf16-mixed",
    "devices": 1,
    "enable_progress_bar": False,
    "ckpt_path": None,
    "log_every_n_steps": 10,
    "input_h5": Path("data/cache/step_2b.h5"),
    "output_h5": Path("data/cache/step_3a.h5"),
    "fast_dev_run": False,
    "overwrite": True,
}


# ------------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------------


class TotalVariationEarlyStopping(Callback):
    """Early stopping based on total variation increase."""

    def __init__(self, patience=3, min_delta=0.0):
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.best_tv = float("inf")
        self.tv_increase_count = 0

    def on_train_epoch_end(self, trainer, pl_module):
        current_tv = trainer.callback_metrics.get("train_tv", None)
        if current_tv is None:
            return
        current_tv = float(current_tv)
        if current_tv > self.best_tv + self.min_delta:
            self.tv_increase_count += 1
            if self.tv_increase_count >= self.patience:
                trainer.should_stop = True
        else:
            self.best_tv = current_tv
            self.tv_increase_count = 0


# ------------------------------------------------------------------
# Dataset — reads from HDF5 step file
# ------------------------------------------------------------------


def _robust_asinh(data: torch.Tensor) -> torch.Tensor:
    """Outlier-robust per-channel normalization (EXPERIMENTAL; opt-in via
    settings['normalization']='robust_asinh'. Default elsewhere is plain z-score,
    so the published pipeline is unchanged).

    Replaces (x-mean)/std -- whose scale a strong observation inflates, suppressing
    weaker modes -- with (x-median)/(1.4826*MAD) then asinh soft-compression. The
    robust scale tracks the noise floor (breakdown point 50% vs std's 0%); asinh
    bounds the strong tail without discarding it. Handles (C,F,T) magnitude and
    (C,F,T,2) complex, normalizing per channel (and component) over the (F,T) axes.
    """
    c = data.shape[0]
    if data.dim() == 3:  # (C, F, T)
        flat = data.reshape(c, -1)
        med = flat.median(dim=1, keepdim=True).values
        mad = (flat - med).abs().median(dim=1, keepdim=True).values
        return torch.asinh((flat - med) / (1.4826 * mad + 1e-6)).reshape(data.shape)
    # (C, F, T, 2): per channel & component
    k = data.shape[-1]
    flat = data.permute(0, 3, 1, 2).reshape(c, k, -1)  # (C, K, F*T)
    med = flat.median(dim=2, keepdim=True).values
    mad = (flat - med).abs().median(dim=2, keepdim=True).values
    z = torch.asinh((flat - med) / (1.4826 * mad + 1e-6))
    return z.reshape(c, k, data.shape[1], data.shape[2]).permute(0, 2, 3, 1)


class SpecDataset(Dataset):
    def __init__(self, h5_path, settings):
        self.h5_path = str(h5_path)
        self.settings = settings
        self._h5 = None

        with h5py.File(self.h5_path, "r") as f:
            self._keys = sorted(f["samples"].keys(), key=int)
        logger.info(f"SpecDataset: {len(self._keys)} samples from {h5_path}")

    def _open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", swmr=True)

    def __len__(self):
        return len(self._keys)

    def __getitem__(self, idx):
        self._open()
        data = torch.from_numpy(np.asarray(self._h5["samples"][self._keys[idx]])).float()
        # data shape: (C, F, T, 2)

        # Auto bin masking
        lower = self.settings.get("_lower_mask_idx", 4)
        upper = self.settings.get("_upper_mask_idx", 3)
        mask_val = self.settings.get("_bin_mask_value", None)
        if mask_val is None:
            mask_val = data.mean().item()
        if lower > 0:
            data[:, :lower] = mask_val
        if upper > 0:
            data[:, -upper:] = mask_val

        # Auto clamp range
        clamp_lo, clamp_hi = self.settings["_clamp_range"]
        data = data.nan_to_num(0, clamp_lo, clamp_hi)
        data = data.clamp(min=clamp_lo, max=clamp_hi)

        norm = self.settings.get("normalization", "zscore")
        if self.settings.get("representation") == "magnitude":
            # collapse (Re, Im) -> magnitude |Z|, then per-channel normalize -> (C, F, T)
            data = torch.sqrt(data[..., 0] ** 2 + data[..., 1] ** 2)
            if norm == "robust_asinh":
                return _robust_asinh(data)
            mean = data.mean(dim=(1, 2), keepdim=True)
            std = data.std(dim=(1, 2), keepdim=True)
            return (data - mean) / (std + 1e-6)

        # Normalize (complex, keeps (C, F, T, 2))
        if norm == "robust_asinh":
            return _robust_asinh(data)
        mean = data.mean(dim=(1, 2), keepdim=True)
        std = data.std(dim=(1, 2), keepdim=True)
        return (data - mean) / (std + 1e-6)


class SpecDataModule(L.LightningDataModule):
    def __init__(self, h5_path, batch_size, num_workers, prefetch_factor, settings):
        super().__init__()
        self.h5_path = h5_path
        self.settings = settings
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

    def setup(self, stage=None):
        self.dataset = SpecDataset(self.h5_path, self.settings)

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            prefetch_factor=self.prefetch_factor,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------


class BTN(nn.Module):
    def __init__(self, in_channels=4, num_layers=5, first_layer_size=32):
        super().__init__()
        self.in_channels = in_channels
        self.unet = UNet(
            in_channels=in_channels,
            out_channels=in_channels,
            num_layers=num_layers,
            first_layer_size=first_layer_size,
        )

    def forward(self, x):
        x_real, x_imag = x[..., 0], -x[..., 1]
        x_real = self.unet(x_real)
        x_imag = self.unet(x_imag)
        return torch.stack([x_real, x_imag], dim=-1)


class BTNMag(nn.Module):
    """Magnitude-only denoiser: single-component U-Net (no real/imag split).

    The ``mag`` ablation feeds the U-Net the biased-Rayleigh magnitude ``|Z|``
    instead of the two zero-mean Gaussian components ``(Re, Im)`` -- this is the
    representation the paper's central claim says breaks the self-supervised
    denoising guarantee. Same width/depth as :class:`BTN` so the contrast is
    representation, not capacity.
    """

    def __init__(self, in_channels=6, num_layers=5, first_layer_size=32):
        super().__init__()
        self.in_channels = in_channels
        self.unet = UNet(
            in_channels=in_channels,
            out_channels=in_channels,
            num_layers=num_layers,
            first_layer_size=first_layer_size,
        )

    def forward(self, x):  # x: (B, in_channels, F, T)
        return self.unet(x)


class BTNModule(L.LightningModule):
    def __init__(
        self,
        adjacent_channels=3,
        total_channels=8,
        num_layers=5,
        first_layer_size=32,
        settings=None,
    ):
        super().__init__()
        if settings is None:
            settings = default_settings
        if adjacent_channels >= total_channels:
            raise ValueError(
                f"adjacent_channels ({adjacent_channels}) must be < total_channels ({total_channels})"
            )
        self.adjacent_channels = adjacent_channels
        self.in_channels = 2 * adjacent_channels
        self.total_channels = total_channels
        self.num_layers = num_layers
        self.first_layer_size = first_layer_size
        self.settings = settings
        self.representation = self.settings.get("representation", "complex")
        if self.representation == "magnitude":
            self.unet = BTNMag(
                in_channels=self.in_channels,
                num_layers=num_layers,
                first_layer_size=first_layer_size,
            )
        else:
            self.unet = BTN(
                in_channels=self.in_channels,
                num_layers=num_layers,
                first_layer_size=first_layer_size,
            )
        if not self.settings.get("fast_dev_run", False) and self.settings.get("compile", True):
            self.unet.compile()

        self.pad_len = self.adjacent_channels
        self.center_channel = self.total_channels // 2 + self.pad_len
        self.padding = nn.ReflectionPad3d((0, 0, 0, 0, self.pad_len, self.pad_len))

        self.loss_fn = nn.L1Loss()
        self.train_tv = TotalVariation()
        self.predict_tv = TotalVariation()

        # HDF5 writer (set in main before predict)
        self._h5_out = None

    def _load_adjacent_channels(self, x, target_channel):
        if target_channel is None:
            target_channel = self.center_channel
        if target_channel < 0:
            target_channel = self.total_channels + target_channel

        channel_idx = target_channel + self.pad_len
        front_channel_idx = channel_idx - self.pad_len
        back_channel_idx = channel_idx + 1 + self.pad_len

        x_real, x_imag = x[..., 0], x[..., 1]
        x_real = self.padding(x_real)
        x_imag = self.padding(x_imag)
        x = torch.stack([x_real, x_imag], dim=-1)

        front_channels = x[:, front_channel_idx:channel_idx]
        back_channels = x[:, channel_idx + 1 : back_channel_idx]
        return torch.cat([front_channels, back_channels], dim=1)

    def _load_target_channels(self, x, target_channel):
        if target_channel is None:
            target_channel = self.center_channel
        if target_channel < 0:
            target_channel = self.total_channels + target_channel
        x = x[:, target_channel : target_channel + 1]
        return x.repeat(1, self.in_channels, 1, 1, 1)

    def _single_channel_loss(self, y_hat, y) -> torch.Tensor:
        loss = self.loss_fn(y_hat, y.flip(-1))
        self.train_tv.update(y_hat[..., 0])
        self.train_tv.update(y_hat[..., 1])
        return loss

    def _multichannel_loss(self, y_hat, y) -> torch.Tensor:
        loss = torch.tensor(0.0, device=y_hat.device)
        for i in range(self.in_channels):
            y_hat_i = y_hat[:, i : i + 1]
            y_i = y[:, i : i + 1]
            loss += self.loss_fn(y_hat_i, y_i.flip(-1))
            self.train_tv.update(y_hat_i[..., 0] / self.in_channels)
            self.train_tv.update(y_hat_i[..., 1] / self.in_channels)
        return loss / self.in_channels

    # --- magnitude-mode variants (representation == "magnitude") ---
    def _load_adjacent_channels_mag(self, x, target_channel):  # x: (B, C, F, T)
        if target_channel is None:
            target_channel = self.center_channel
        if target_channel < 0:
            target_channel = self.total_channels + target_channel
        channel_idx = target_channel + self.pad_len
        front_channel_idx = channel_idx - self.pad_len
        back_channel_idx = channel_idx + 1 + self.pad_len
        xp = self.padding(x)
        front_channels = xp[:, front_channel_idx:channel_idx]
        back_channels = xp[:, channel_idx + 1 : back_channel_idx]
        return torch.cat([front_channels, back_channels], dim=1)

    def _load_target_channels_mag(self, x, target_channel):  # x: (B, C, F, T)
        if target_channel is None:
            target_channel = self.center_channel
        if target_channel < 0:
            target_channel = self.total_channels + target_channel
        x = x[:, target_channel : target_channel + 1]
        return x.repeat(1, self.in_channels, 1, 1)

    def _training_step_mag(self, batch, batch_idx):
        target_channel = torch.randint(0, self.total_channels, (1,))
        loss = torch.tensor(0.0, device=batch.device)

        x1 = self._load_adjacent_channels_mag(batch, target_channel)
        y1 = batch[:, target_channel : target_channel + 1]
        y_hat1 = self(x1).mean(dim=1, keepdim=True)
        loss1 = self.loss_fn(y_hat1, y1)
        self.train_tv.update(y_hat1)
        loss += loss1
        self.log("train_loss1", loss1)

        x2 = self._load_target_channels_mag(batch, target_channel)
        y_hat2 = self(x2)
        loss2 = torch.tensor(0.0, device=batch.device)
        for i in range(self.in_channels):
            y_hat_i = y_hat2[:, i : i + 1]
            y_i = x1[:, i : i + 1]
            loss2 = loss2 + self.loss_fn(y_hat_i, y_i)
            self.train_tv.update(y_hat_i / self.in_channels)
        loss2 = loss2 / self.in_channels
        loss += loss2
        self.log("train_loss2", loss2)
        self.log("train_loss", loss)
        return loss

    def _predict_step_mag(self, batch, batch_idx):
        target_channels = torch.arange(self.total_channels)
        xs = [self._load_adjacent_channels_mag(batch, tc) for tc in target_channels]
        ys = [self._load_target_channels_mag(batch, tc) for tc in target_channels]
        y_hats1 = torch.cat([self(x).mean(dim=1, keepdim=True) for x in xs], dim=1)
        y_hats2 = torch.cat([self(y).mean(dim=1, keepdim=True) for y in ys], dim=1)
        # store as (B, C, F, T, 2): two magnitude estimates -> step_4a RMS-combines them
        out_data = torch.stack([y_hats1, y_hats2], dim=-1).float().cpu().numpy()
        if self._h5_out is not None:
            write_sample(self._h5_out, batch_idx, out_data)
        self.predict_tv.update(y_hats1)
        self.predict_tv.update(y_hats2)
        return 0

    def forward(self, x):
        return self.unet(x)

    def training_step(self, batch, batch_idx):
        if self.representation == "magnitude":
            return self._training_step_mag(batch, batch_idx)
        target_channel = torch.randint(0, self.total_channels, (1,))
        loss = torch.tensor(0.0, device=batch.device)

        x1 = self._load_adjacent_channels(batch, target_channel)
        y1 = batch[:, target_channel : target_channel + 1]
        y_hat1 = self(x1).mean(dim=1).unsqueeze(1)
        loss1 = self._single_channel_loss(y_hat1, y1)
        loss += loss1
        self.log("train_loss1", loss1)

        x2 = self._load_target_channels(batch, target_channel)
        y_hat2 = self(x2)
        loss2 = self._multichannel_loss(y_hat2, x1)
        loss += loss2
        self.log("train_loss2", loss2)
        self.log("train_loss", loss)
        return loss

    def on_train_epoch_end(self):
        tv = self.train_tv.compute()
        self.log("train_tv", tv, logger=True)
        self.train_tv.reset()

    def predict_step(self, batch, batch_idx):
        if self.representation == "magnitude":
            return self._predict_step_mag(batch, batch_idx)
        target_channels = torch.arange(self.total_channels)

        xs = [self._load_adjacent_channels(batch, tc) for tc in target_channels]
        ys = [self._load_target_channels(batch, tc) for tc in target_channels]

        y_hats1 = torch.cat([self(x).mean(dim=1).unsqueeze(1) for x in xs], dim=1)
        y_hats2 = torch.cat([self(y).mean(dim=1).unsqueeze(1) for y in ys], dim=1)

        out_data = torch.cat([y_hats1, y_hats2], dim=-1).float().cpu().numpy()

        # Write to HDF5
        if self._h5_out is not None:
            write_sample(self._h5_out, batch_idx, out_data)

        self.predict_tv.update(y_hats1[..., 0])
        self.predict_tv.update(y_hats1[..., 1])
        self.predict_tv.update(y_hats2[..., 0])
        self.predict_tv.update(y_hats2[..., 1])
        return 0

    def on_predict_epoch_end(self):
        tv = self.predict_tv.compute()
        logger.info(f"Predict TV: {tv}")
        self.predict_tv.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main(
    config_path: Path | str | None = None,
    settings: dict | None = None,
) -> None:
    if settings is None:
        settings = load_settings(config_path, default_settings)

    input_h5 = Path(settings.get("input_h5", default_settings["input_h5"]))
    output_h5 = Path(settings.get("output_h5", default_settings["output_h5"]))
    output_h5.parent.mkdir(parents=True, exist_ok=True)

    # --- Auto clamp range ---
    clamp_range = settings.get("clamp_range", "auto")
    if clamp_range == "auto":
        pcts = settings.get("clamp_percentiles", [1, 99])
        settings["_clamp_range"] = compute_clamp_range(input_h5, tuple(pcts))
    else:
        settings["_clamp_range"] = tuple(clamp_range)

    # --- Auto bin masking ---
    bin_masking = settings.get("bin_masking", "auto")
    if bin_masking == "auto":
        gt = settings.get("gradient_threshold", 0.5)
        lo, hi = detect_edge_bins(input_h5, gradient_threshold=gt)
        settings["_lower_mask_idx"] = lo
        settings["_upper_mask_idx"] = hi
    else:
        settings["_lower_mask_idx"] = int(bin_masking)
        settings["_upper_mask_idx"] = int(bin_masking)

    mask_val_cfg = settings.get("bin_mask_value", "mean")
    settings["_bin_mask_value"] = None if mask_val_cfg == "mean" else float(mask_val_cfg)

    # --- Build data module ---
    datamodule = SpecDataModule(
        h5_path=input_h5,
        batch_size=settings.get("batch_size", 36),
        num_workers=settings.get("num_workers", 8),
        prefetch_factor=settings.get("prefetch_factor", 2),
        settings=settings,
    )

    # --- Build model ---
    model = BTNModule(
        adjacent_channels=settings.get("adjacent_channels", default_settings["adjacent_channels"]),
        total_channels=settings.get("total_channels", default_settings["total_channels"]),
        num_layers=settings.get("num_layers", default_settings["num_layers"]),
        first_layer_size=settings.get("first_layer_size", default_settings["first_layer_size"]),
        settings=settings,
    )

    callbacks = []
    if settings.get("tv_early_stopping", True):
        callbacks.append(TotalVariationEarlyStopping(patience=settings.get("tv_patience", 3)))

    trainer = L.Trainer(
        max_epochs=settings.get("max_epochs", 30),
        precision=settings.get("precision", "bf16-mixed"),
        devices=settings.get("devices", 1),
        log_every_n_steps=settings.get("log_every_n_steps", 10),
        enable_progress_bar=settings.get("enable_progress_bar", False),
        fast_dev_run=settings.get("fast_dev_run", False),
        callbacks=callbacks,
    )

    ckpt = settings.get("ckpt_path")
    if ckpt is not None:
        trainer.fit(model, datamodule, ckpt_path=ckpt)
    else:
        trainer.fit(model, datamodule)

    # Predict → write to HDF5
    h5 = create_step_file(output_h5, metadata={
        "clamp_range": list(settings["_clamp_range"]),
        "lower_mask_idx": settings["_lower_mask_idx"],
        "upper_mask_idx": settings["_upper_mask_idx"],
        "first_layer_size": settings.get("first_layer_size", 32),
    })
    model._h5_out = h5
    try:
        trainer.predict(model, datamodule)
    finally:
        h5.close()

    logger.info(f"Wrote predictions to {output_h5}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path)
