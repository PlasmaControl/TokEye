"""step_7 — final segmenter trained on soft targets, exported for deployment.

The training target is the shrinkage blend of the knee labels and the
out-of-fold model evidence from step_6:

    q = (1 - model_trust) * y0 + model_trust * p_oof

trained with BCE (soft bootstrapping): BCE's minimizer is E[q], so the model
learns the smoothed consensus. model_trust = 0 reproduces training directly
on the knee labels (no refine); no extra label smoothing is applied (q is
already soft). Unlike the old never-drop-positives heuristic this is
symmetric — confident model disagreement can remove knee false positives.

Exports to ``model_dir``: best checkpoint, ``final.torchscript.pt`` (traced
from a fresh uncompiled UNet), and ``deploy_manifest.yaml`` carrying
everything inference needs: scale (nfft/hop), per-modality robust stats and
``a`` from step_5, and the edge bins from the resolved-params ledger.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import h5py
import lightning as L
import numpy as np
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import Dataset, Subset

from tokeye.models.modules.unet import UNet

from .step_6_refine import _TRAIN_DEFAULTS, RefineModule, _loader
from .utils.augmentations import get_augmentation

if TYPE_CHECKING:
    from pathlib import Path

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True

logger = logging.getLogger(__name__)


class SoftTargetDataset(Dataset):
    """(image, q) pairs: images from step_5, q blended on the fly."""

    def __init__(
        self,
        dataset_h5: Path,
        refine_h5: Path,
        model_trust: float,
        transform=None,
    ) -> None:
        self.dataset_h5 = dataset_h5
        self.refine_h5 = refine_h5
        self.model_trust = model_trust
        self.transform = transform
        with h5py.File(dataset_h5, "r") as f:
            self.n = int(f.attrs["n_samples"])
        self._data: h5py.File | None = None
        self._refine: h5py.File | None = None

    def __len__(self) -> int:
        return self.n

    def _files(self) -> tuple[h5py.File, h5py.File]:
        if self._data is None:
            self._data = h5py.File(self.dataset_h5, "r", swmr=True)
            self._refine = h5py.File(self.refine_h5, "r", swmr=True)
        return self._data, self._refine

    def __getitem__(self, idx: int):
        data, refine = self._files()
        x = torch.from_numpy(np.asarray(data["images"][str(idx)])).float()
        y0 = torch.from_numpy(np.asarray(data["masks"][str(idx)])).float()
        lam = self.model_trust
        if lam > 0:
            p = torch.from_numpy(np.asarray(refine["p_oof"][idx])).float()
            q = (1 - lam) * y0 + lam * p
        else:
            q = y0
        if self.transform is not None:
            x, q = self.transform(x, q)
        return x, q


def main(settings: dict) -> None:
    L.seed_everything(42)
    model_dir = settings["model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)

    train_settings = {**settings, "loss_type": settings["loss_type"]}
    train_ds = SoftTargetDataset(
        settings["dataset_h5"],
        settings["refine_h5"],
        settings["model_trust"],
        transform=get_augmentation(_TRAIN_DEFAULTS),
    )
    val_ds = SoftTargetDataset(
        settings["dataset_h5"],
        settings["refine_h5"],
        settings["model_trust"],
        transform=None,
    )
    n = len(train_ds)
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    n_val = max(1, int(0.1 * n))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    module = RefineModule(train_settings)
    ckpt = ModelCheckpoint(dirpath=model_dir, monitor="val_loss", save_top_k=1)
    trainer = L.Trainer(
        max_epochs=settings["max_epochs"],
        precision=settings["precision"],
        accelerator="auto",
        devices=1,
        callbacks=[ckpt, EarlyStopping(monitor="val_loss", patience=5)],
        enable_progress_bar=False,
        log_every_n_steps=10,
        logger=False,
    )
    trainer.fit(
        module,
        train_dataloaders=_loader(Subset(train_ds, train_idx), settings, shuffle=True),
        val_dataloaders=_loader(Subset(val_ds, val_idx), settings, shuffle=False),
    )
    logger.info(f"best checkpoint: {ckpt.best_model_path}")

    # TorchScript export: trace a FRESH uncompiled UNet (compiled modules
    # cannot be traced), loading the best weights.
    best = RefineModule.load_from_checkpoint(
        ckpt.best_model_path, settings={**train_settings, "smoke": True}
    )
    export_net = UNet(
        in_channels=1,
        out_channels=2,
        num_layers=settings["num_layers"],
        first_layer_size=settings["first_layer_size"],
    )
    export_net.load_state_dict(best.unet.state_dict())
    export_net.eval()
    sample_x, _ = val_ds[0]
    example = sample_x[None]  # (1, 1, F, T)
    with torch.no_grad():
        traced = torch.jit.trace(export_net, example)
    ts_path = model_dir / "final.torchscript.pt"
    traced.save(str(ts_path))
    logger.info(f"TorchScript -> {ts_path}")

    # Deploy manifest: everything inference needs to reproduce the input
    # normalization at this scale.
    with h5py.File(settings["dataset_h5"], "r") as f:
        stats = {
            k: float(v)
            for k, v in f.attrs.items()
            if k.endswith(("_median", "_scale"))
        }
        a = float(f.attrs["a"])
    resolved = {}
    if settings["resolved_params"].exists():
        resolved = yaml.safe_load(settings["resolved_params"].read_text()) or {}
    edge_bins = {
        mod: {k: entry[k]["value"] for k in entry if k.startswith("edge_bins")}
        for mod, entry in resolved.get("step_2", {}).items()
    }
    manifest = {
        "run_id": settings["run_id"],
        "nfft": settings["nfft"],
        "hop": settings["hop"],
        "normalization": {"form": "a*asinh((log1p|Z| - median)/(a*scale))", "a": a},
        "modality_stats": stats,
        "edge_bins": edge_bins,
        "model_trust": settings["model_trust"],
        "checkpoint": str(ckpt.best_model_path),
        "torchscript": str(ts_path),
    }
    settings["deploy_manifest"].write_text(yaml.safe_dump(manifest, sort_keys=False))
    logger.info(f"manifest -> {settings['deploy_manifest']}")
