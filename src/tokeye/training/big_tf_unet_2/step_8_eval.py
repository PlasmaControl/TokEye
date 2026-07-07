"""step_8 — quick TJ-II benchmark: one meaningful number per run.

Runs the exported TorchScript model on the public TJ-II 2021 set
(``data/eval/TJII2021``) with the SAME preprocessing as the paper eval
(vertical flip, the TJ-II pixel mean/std, reflect-pad to /32), sweeps the
sigmoid threshold, and reports per-image IoU + F1 at the F1-optimal
threshold. The deployed big_tf_unet anchor on this metric is IoU 0.260 /
F1 0.478 (F1-opt threshold 0.80).

Caveat printed with the result: the TJ-II images are fixed-resolution
pre-rendered spectrograms, so a teacher trained at a different (nfft, hop)
sees a mild domain shift here — treat the number as a relative tracker
across runs, not an absolute target.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# TJ-II pixel statistics used by the paper eval (dev/paper/eval/TJII2021*.py).
_TJII_MEAN = 17.84620821169868
_TJII_STD = 25.016818830630463
_ANCHOR = {"iou": 0.260, "f1": 0.478, "threshold": 0.80}


def _load_images(dataset_dir):
    """Yield (spec (1,1,H,W) float, gt (H,W) bool) per TJ-II shot."""
    for spec_path in sorted((dataset_dir / "input").glob("*.png")):
        gt_path = dataset_dir / "gt" / spec_path.name
        if not gt_path.exists():
            continue
        spec = np.array(Image.open(spec_path).convert("L"))
        ann = np.array(Image.open(gt_path).convert("L"))
        if spec.shape != ann.shape:  # one malformed shot in the public set
            logger.warning(f"skipping {spec_path.name}: size mismatch")
            continue
        spec = np.flip(spec, axis=0).copy()
        ann = np.flip(ann, axis=0).copy()
        spec = (spec - _TJII_MEAN) / _TJII_STD
        yield (
            torch.from_numpy(spec).float()[None, None],
            (ann // 255).astype(bool),
        )


def _pad32(x: torch.Tensor) -> torch.Tensor:
    h, w = x.shape[-2:]
    ph, pw = (-h) % 32, (-w) % 32
    if ph or pw:
        x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
    return x


def main(settings: dict) -> None:
    ts_path = settings["model_dir"] / "final.torchscript.pt"
    if not ts_path.exists():
        raise FileNotFoundError(f"no exported model at {ts_path} — run step_7 first")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.jit.load(str(ts_path), map_location=device).eval()

    thresholds = np.linspace(0.02, 0.98, settings["n_thresholds"])
    tp = np.zeros(len(thresholds))
    fp = np.zeros(len(thresholds))
    fn = np.zeros(len(thresholds))
    iou_per_image: list[np.ndarray] = []

    n_images = 0
    with torch.no_grad():
        for spec, gt in _load_images(settings["dataset_dir"]):
            h, w = spec.shape[-2:]
            out = model(_pad32(spec).to(device))
            if isinstance(out, (tuple, list)):
                out = out[0]
            prob = torch.sigmoid(out[:, 0:1])[..., :h, :w].cpu().numpy()[0, 0]
            gt_sum = gt.sum()
            image_iou = np.zeros(len(thresholds))
            for i, t in enumerate(thresholds):
                pred = prob > t
                inter = np.logical_and(pred, gt).sum()
                pred_sum = pred.sum()
                tp[i] += inter
                fp[i] += pred_sum - inter
                fn[i] += gt_sum - inter
                union = pred_sum + gt_sum - inter
                image_iou[i] = inter / union if union > 0 else 1.0
            iou_per_image.append(image_iou)
            n_images += 1

    if n_images == 0:
        raise RuntimeError(f"no evaluable images in {settings['dataset_dir']}")

    iou_matrix = np.stack(iou_per_image)  # (n_images, n_thresholds)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp + fn, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = int(np.argmax(f1))

    rows = pd.DataFrame(
        {
            "threshold": thresholds,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou_per_image_mean": iou_matrix.mean(axis=0),
        }
    )
    rows.to_csv(settings["out_csv"], index=False)

    iou_opt = float(iou_matrix.mean(axis=0)[best])
    logger.info(
        f"TJ-II ({n_images} images): IoU={iou_opt:.3f}, F1={f1[best]:.3f} "
        f"at threshold {thresholds[best]:.2f} "
        f"(deployed anchor: IoU={_ANCHOR['iou']}, F1={_ANCHOR['f1']} "
        f"at {_ANCHOR['threshold']})"
    )
    logger.info(
        "note: TJ-II images are fixed-resolution renders — for a per-scale "
        "teacher this number is a relative tracker across runs, not a target"
    )
    logger.info(f"sweep -> {settings['out_csv']}")
