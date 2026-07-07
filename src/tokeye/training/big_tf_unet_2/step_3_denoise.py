"""step_3 — self-supervised cross-channel denoising (train + per-sample extract).

Trains a U-Net to predict each channel's residual spectrogram from its
``adjacent_channels`` neighbors: sensor noise is independent across channels
while coherent modes are shared, so the network can only reproduce the shared
signal. After training, prediction runs over the full dataset in order and each
sample's denoised field is written to ``out_h5`` keyed by its original index —
the ablation pipeline's separate step_3b unpack is folded into this step.

Preprocessing per sample (input ``(C, F, T, 2)`` step_2 residual):

- ``representation="magnitude"`` collapses (Re, Im) -> |Z| before normalizing.
- ``normalization="robust_asinh"``: N_a(x) = a * asinh((x - med) / (a * 1.4826
  * MAD)) per channel (and per real/imag component); ``a=1`` reproduces the
  ablation's robust-asinh exactly.
- ``normalization="zscore"``: plain per-sample per-channel z-score.
- The bottom ``edge_bins_lower`` and top ``edge_bins_upper`` frequency rows are
  zeroed AFTER normalization (0 is the post-N_a median).

Output ``/samples/{idx}`` (float32, same indexing as ``in_h5``): the mean of
the model's two denoised estimates (adjacent-channel prediction and
self-reconstruction) — ``(C, F, T, 2)`` real/imag for the complex
representation, ``(C, F, T)`` for magnitude.
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import lightning as L
import torch
from lightning.pytorch.callbacks import BasePredictionWriter, Callback
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.image import TotalVariation

from tokeye.models.modules.unet import UNet

from .utils.hdf5_io import (
    HDF5StepDataset,
    create_step_file,
    worker_init_fn,
    write_sample,
)

# TF32: fp32 storage + tensor-core matmul/conv (10-bit operands, fp32
# accumulate) -- ~bf16 speed, ~fp32 stability. MUST use the legacy API:
# Lightning calls torch.get_float32_matmul_precision() internally, so the new
# backends.*.fp32_precision API raises "mix of legacy and new APIs" at Trainer
# setup.
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Normalization
# ------------------------------------------------------------------


def _robust_asinh(data: torch.Tensor, a: float) -> torch.Tensor:
    """N_a(x) = a * asinh((x - median) / (a * 1.4826 * MAD)), per channel.

    The robust scale tracks the noise floor (breakdown point 50% vs std's 0%),
    so a strong observation cannot inflate the divisor and suppress weaker
    modes; asinh soft-bounds the strong tail without discarding it, with ``a``
    setting where compression kicks in (``a=1`` reproduces the ablation's
    robust-asinh exactly). Handles ``(C, F, T)`` magnitude and ``(C, F, T, 2)``
    complex, normalizing per channel (and component) over the (F, T) axes.
    """
    c = data.shape[0]
    if data.dim() == 3:  # (C, F, T)
        flat = data.reshape(c, -1)
        med = flat.median(dim=1, keepdim=True).values
        mad = (flat - med).abs().median(dim=1, keepdim=True).values
        z = a * torch.asinh((flat - med) / (a * (1.4826 * mad + 1e-6)))
        return z.reshape(data.shape)
    # (C, F, T, 2): per channel & component
    k = data.shape[-1]
    flat = data.permute(0, 3, 1, 2).reshape(c, k, -1)  # (C, K, F*T)
    med = flat.median(dim=2, keepdim=True).values
    mad = (flat - med).abs().median(dim=2, keepdim=True).values
    z = a * torch.asinh((flat - med) / (a * (1.4826 * mad + 1e-6)))
    return z.reshape(c, k, data.shape[1], data.shape[2]).permute(0, 2, 3, 1)


class Preprocess:
    """Per-sample transform: representation collapse, normalize, zero edges.

    Module-level class (not a closure) so DataLoader workers can pickle it.
    """

    def __init__(
        self,
        representation: str,
        normalization: str,
        a: float,
        edge_bins_lower: int,
        edge_bins_upper: int,
    ) -> None:
        self.representation = representation
        self.normalization = normalization
        self.a = a
        self.edge_bins_lower = edge_bins_lower
        self.edge_bins_upper = edge_bins_upper

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        data = torch.nan_to_num(data)
        if self.representation == "magnitude":
            # collapse (Re, Im) -> |Z|: (C, F, T, 2) -> (C, F, T)
            data = torch.sqrt(data[..., 0] ** 2 + data[..., 1] ** 2)
        if self.normalization == "robust_asinh":
            data = _robust_asinh(data, self.a)
        else:  # per-sample per-channel z-score (per component for complex)
            mean = data.mean(dim=(1, 2), keepdim=True)
            std = data.std(dim=(1, 2), keepdim=True)
            data = (data - mean) / (std + 1e-6)
        # Edge-bin masking after normalization: 0 is the post-N_a median.
        if self.edge_bins_lower > 0:
            data[:, : self.edge_bins_lower] = 0.0
        if self.edge_bins_upper > 0:
            data[:, -self.edge_bins_upper :] = 0.0
        return data


# ------------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------------


class TotalVariationEarlyStopping(Callback):
    """Early stopping based on total variation increase."""

    def __init__(self, patience: int = 3, min_delta: float = 0.0) -> None:
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.best_tv = float("inf")
        self.tv_increase_count = 0

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
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


class DenoisedSampleWriter(BasePredictionWriter):
    """Streams each denoised sample to the step HDF5 as batches complete.

    Keeps the file open across batches (opened at predict start) and keys
    every sample by its original ``in_h5`` index — the predict loader is
    sequential, so a running position maps 1:1 onto the sorted sample keys.
    Predictions never accumulate in memory.
    """

    def __init__(
        self,
        out_h5: Path,
        sample_keys: list[str],
        metadata: dict | None = None,
    ) -> None:
        super().__init__(write_interval="batch")
        self.out_h5 = out_h5
        self.sample_keys = sample_keys
        self.metadata = metadata
        self._h5: h5py.File | None = None
        self._pos = 0

    def on_predict_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        self._h5 = create_step_file(self.out_h5, metadata=self.metadata)
        self._pos = 0

    def write_on_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        prediction: torch.Tensor,
        batch_indices,
        batch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        for sample in prediction.numpy():
            write_sample(self._h5, int(self.sample_keys[self._pos]), sample)
            self._pos += 1

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.attrs["num_samples"] = self._pos
            self._h5.close()
            self._h5 = None


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------


class BTN(nn.Module):
    """Complex-field denoiser: one shared U-Net over the (Re, -Im) components.

    Negating the imaginary part means the net sees the conjugate; paired with
    the ``y.flip(-1)`` training target this ties the two zero-mean Gaussian
    components together instead of treating them as unrelated planes.
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_layers: int = 5,
        first_layer_size: int = 32,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.unet = UNet(
            in_channels=in_channels,
            out_channels=in_channels,
            num_layers=num_layers,
            first_layer_size=first_layer_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 2A, F, T, 2)
        x_real, x_imag = x[..., 0], -x[..., 1]
        return torch.stack([self.unet(x_real), self.unet(x_imag)], dim=-1)


class BTNMag(nn.Module):
    """Magnitude-only denoiser: single-component U-Net (no real/imag split).

    Same width/depth as :class:`BTN` so the contrast between representations
    is representation, not capacity.
    """

    def __init__(
        self,
        in_channels: int = 6,
        num_layers: int = 5,
        first_layer_size: int = 32,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.unet = UNet(
            in_channels=in_channels,
            out_channels=in_channels,
            num_layers=num_layers,
            first_layer_size=first_layer_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 2A, F, T)
        return self.unet(x)


class DenoiseModule(L.LightningModule):
    """Self-supervised cross-channel denoiser (L1, TV-tracked).

    Two training paths per batch: predict a random target channel from its
    ``adjacent_channels`` neighbors, and reconstruct each neighbor from the
    target channel repeated. Prediction averages the two estimates per channel.
    """

    def __init__(
        self,
        representation: str = "complex",
        adjacent_channels: int = 3,
        total_channels: int = 8,
        num_layers: int = 5,
        first_layer_size: int = 32,
        compile_model: bool = False,
    ) -> None:
        super().__init__()
        if adjacent_channels >= total_channels:
            raise ValueError(
                f"adjacent_channels ({adjacent_channels}) must be < "
                f"total_channels ({total_channels})"
            )
        self.representation = representation
        self.adjacent_channels = adjacent_channels
        self.in_channels = 2 * adjacent_channels
        self.total_channels = total_channels
        net_cls = BTNMag if representation == "magnitude" else BTN
        self.net = net_cls(
            in_channels=self.in_channels,
            num_layers=num_layers,
            first_layer_size=first_layer_size,
        )
        if compile_model:
            self.net.compile()

        self.pad_len = self.adjacent_channels
        # Pads the channel axis: dim -3 of (B, C, F, T) inputs.
        self.padding = nn.ReflectionPad3d((0, 0, 0, 0, self.pad_len, self.pad_len))

        self.loss_fn = nn.L1Loss()
        self.train_tv = TotalVariation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    # --- channel gather (complex, (B, C, F, T, 2)) ---

    def _load_adjacent_channels(self, x: torch.Tensor, target: int) -> torch.Tensor:
        """The 2A channels around ``target`` -> (B, 2A, F, T, 2)."""
        channel_idx = target + self.pad_len
        x_real, x_imag = x[..., 0], x[..., 1]
        xp = torch.stack([self.padding(x_real), self.padding(x_imag)], dim=-1)
        front = xp[:, channel_idx - self.pad_len : channel_idx]
        back = xp[:, channel_idx + 1 : channel_idx + 1 + self.pad_len]
        return torch.cat([front, back], dim=1)

    def _load_target_channels(self, x: torch.Tensor, target: int) -> torch.Tensor:
        """``target`` repeated to the net's input width -> (B, 2A, F, T, 2)."""
        return x[:, target : target + 1].repeat(1, self.in_channels, 1, 1, 1)

    # --- channel gather (magnitude, (B, C, F, T)) ---

    def _load_adjacent_channels_mag(
        self, x: torch.Tensor, target: int
    ) -> torch.Tensor:
        channel_idx = target + self.pad_len
        xp = self.padding(x)
        front = xp[:, channel_idx - self.pad_len : channel_idx]
        back = xp[:, channel_idx + 1 : channel_idx + 1 + self.pad_len]
        return torch.cat([front, back], dim=1)

    def _load_target_channels_mag(
        self, x: torch.Tensor, target: int
    ) -> torch.Tensor:
        return x[:, target : target + 1].repeat(1, self.in_channels, 1, 1)

    # --- losses (complex) ---

    def _single_channel_loss(
        self, y_hat: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        loss = self.loss_fn(y_hat, y.flip(-1))
        self.train_tv.update(y_hat[..., 0])
        self.train_tv.update(y_hat[..., 1])
        return loss

    def _multichannel_loss(
        self, y_hat: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=y_hat.device)
        for i in range(self.in_channels):
            y_hat_i = y_hat[:, i : i + 1]
            y_i = y[:, i : i + 1]
            loss = loss + self.loss_fn(y_hat_i, y_i.flip(-1))
            self.train_tv.update(y_hat_i[..., 0] / self.in_channels)
            self.train_tv.update(y_hat_i[..., 1] / self.in_channels)
        return loss / self.in_channels

    # --- training ---

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        if self.representation == "magnitude":
            return self._training_step_mag(batch)
        target = int(torch.randint(0, self.total_channels, (1,)))

        x1 = self._load_adjacent_channels(batch, target)
        y1 = batch[:, target : target + 1]
        y_hat1 = self(x1).mean(dim=1, keepdim=True)
        loss1 = self._single_channel_loss(y_hat1, y1)
        self.log("train_loss1", loss1)

        x2 = self._load_target_channels(batch, target)
        loss2 = self._multichannel_loss(self(x2), x1)
        self.log("train_loss2", loss2)

        loss = loss1 + loss2
        self.log("train_loss", loss)
        return loss

    def _training_step_mag(self, batch: torch.Tensor) -> torch.Tensor:
        target = int(torch.randint(0, self.total_channels, (1,)))

        x1 = self._load_adjacent_channels_mag(batch, target)
        y1 = batch[:, target : target + 1]
        y_hat1 = self(x1).mean(dim=1, keepdim=True)
        loss1 = self.loss_fn(y_hat1, y1)
        self.train_tv.update(y_hat1)
        self.log("train_loss1", loss1)

        x2 = self._load_target_channels_mag(batch, target)
        y_hat2 = self(x2)
        loss2 = torch.tensor(0.0, device=batch.device)
        for i in range(self.in_channels):
            y_hat_i = y_hat2[:, i : i + 1]
            loss2 = loss2 + self.loss_fn(y_hat_i, x1[:, i : i + 1])
            self.train_tv.update(y_hat_i / self.in_channels)
        loss2 = loss2 / self.in_channels
        self.log("train_loss2", loss2)

        loss = loss1 + loss2
        self.log("train_loss", loss)
        return loss

    def on_train_epoch_end(self) -> None:
        self.log("train_tv", self.train_tv.compute(), logger=True)
        self.train_tv.reset()

    # --- prediction ---

    def _denoise_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Denoise every channel of a batch -> (B, C, F, T[, 2])."""
        targets = range(self.total_channels)
        if self.representation == "magnitude":
            xs = [self._load_adjacent_channels_mag(batch, t) for t in targets]
            ys = [self._load_target_channels_mag(batch, t) for t in targets]
            y_hats1 = torch.cat(
                [self(x).mean(dim=1, keepdim=True) for x in xs], dim=1
            )
            y_hats2 = torch.cat(
                [self(y).mean(dim=1, keepdim=True) for y in ys], dim=1
            )
            return (y_hats1 + y_hats2) / 2
        xs = [self._load_adjacent_channels(batch, t) for t in targets]
        ys = [self._load_target_channels(batch, t) for t in targets]
        y_hats1 = torch.cat([self(x).mean(dim=1, keepdim=True) for x in xs], dim=1)
        y_hats2 = torch.cat([self(y).mean(dim=1, keepdim=True) for y in ys], dim=1)
        # The complex net is trained against y.flip(-1) (conjugate trick), so
        # the estimates come out (Im, Re); flip back to (Re, Im).
        return ((y_hats1 + y_hats2) / 2).flip(-1)

    def predict_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._denoise_batch(batch).float().cpu()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=1e-4)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def _make_loader(
    dataset: HDF5StepDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    """DataLoader with the efficiency flags, guarded when num_workers == 0."""
    kwargs: dict = {}
    if num_workers > 0:
        kwargs.update(
            persistent_workers=True,
            prefetch_factor=2,
            worker_init_fn=worker_init_fn,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        **kwargs,
    )


def main(settings: dict) -> dict | None:
    in_h5 = Path(settings["in_h5"])
    out_h5 = Path(settings["out_h5"])
    representation = str(settings["representation"])
    normalization = str(settings["normalization"])
    a = float(settings["a"])
    edge_lower = int(settings["edge_bins_lower"])
    edge_upper = int(settings["edge_bins_upper"])
    batch_size = int(settings["batch_size"])
    num_workers = int(settings["num_workers"])
    smoke = bool(settings["smoke"])

    L.seed_everything(42)

    dataset = HDF5StepDataset(
        in_h5,
        transform=Preprocess(
            representation=representation,
            normalization=normalization,
            a=a,
            edge_bins_lower=edge_lower,
            edge_bins_upper=edge_upper,
        ),
    )
    with h5py.File(in_h5, "r") as f:
        sample_keys = sorted(f["samples"].keys(), key=int)
    logger.info(
        f"Denoising {len(dataset)} samples from {in_h5} "
        f"(representation={representation}, normalization={normalization}, "
        f"a={a:g}, edges: lower={edge_lower}, upper={edge_upper})"
    )

    model = DenoiseModule(
        representation=representation,
        adjacent_channels=int(settings["adjacent_channels"]),
        total_channels=int(settings["total_channels"]),
        num_layers=int(settings["num_layers"]),
        first_layer_size=int(settings["first_layer_size"]),
        compile_model=not smoke and torch.cuda.is_available(),
    )

    writer = DenoisedSampleWriter(
        out_h5,
        sample_keys,
        metadata={
            "run_id": settings["run_id"],
            "representation": representation,
            "normalization": normalization,
            "a": a,
            "edge_bins_lower": edge_lower,
            "edge_bins_upper": edge_upper,
            "first_layer_size": int(settings["first_layer_size"]),
        },
    )
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=int(settings["max_epochs"]),
        precision=settings["precision"],
        log_every_n_steps=10,
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
        callbacks=[
            TotalVariationEarlyStopping(patience=int(settings["tv_patience"])),
            writer,
        ],
    )

    trainer.fit(model, _make_loader(dataset, batch_size, num_workers, shuffle=True))

    try:
        trainer.predict(
            model,
            dataloaders=_make_loader(dataset, batch_size, num_workers, shuffle=False),
            return_predictions=False,
        )
    finally:
        writer.close()

    logger.info(f"Wrote denoised samples to {out_h5}")
    return None
