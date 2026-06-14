"""Model-based window activity filter.

Runs the existing full-recipe TokEye surrogate on each window's per-channel
magnitude spectrogram and keeps the most-active windows per shot. Applied
identically to all variants, so it cannot bias the relative comparison -- it
only selects *which* windows everyone trains on, not the labels.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _remap_legacy_state_dict(sd: dict) -> dict:
    idx_map = {"0": "0", "1": "1", "4": "3", "5": "4"}
    out = {}
    for k, v in sd.items():
        nk = k.replace(".double_conv.", ".conv.").replace(".maxpool_conv.1.", ".down.1.")
        parts = nk.split(".")
        for i, p in enumerate(parts):
            if p == "conv" and i + 1 < len(parts) and parts[i + 1] in idx_map:
                parts[i + 1] = idx_map[parts[i + 1]]
                break
        out[".".join(parts)] = v
    return out


def load_filter_model(weights: str | Path, fallback: str | Path | None, device: str):
    from tokeye.models.big_tf_unet.config_big_tf_unet import BigTFUNetConfig
    from tokeye.models.big_tf_unet.model_big_tf_unet import BigTFUNetModel

    wp = Path(weights)
    if not wp.exists() and fallback is not None:
        wp = Path(fallback)
    cfg = BigTFUNetConfig(
        in_channels=1, out_channels=2, num_layers=5, first_layer_size=32, dropout_rate=0.0
    )
    model = BigTFUNetModel(cfg)
    sd = _remap_legacy_state_dict(torch.load(wp, weights_only=True, map_location="cpu"))
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    logger.info(f"filter model loaded from {wp}")
    return model


def _pad_to_multiple(x: torch.Tensor, m: int = 32) -> torch.Tensor:
    """Pad (1,1,H,W) so H,W are multiples of m (reflect padding)."""
    h, w = x.shape[-2:]
    ph, pw = (-h) % m, (-w) % m
    if ph or pw:
        x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
    return x


def activity_score_from_sigmoid(sig: np.ndarray, threshold: float) -> float:
    """sig: (2, H, W) sigmoid outputs -> max over channels of active-pixel fraction."""
    active = (sig > threshold).mean(axis=(1, 2))  # per channel fraction
    return float(active.max())


@torch.no_grad()
def score_window(
    model,
    complex_window: np.ndarray,
    mean: float,
    std: float,
    threshold: float,
    device: str,
) -> float:
    """complex_window: (C, F, T, 2). Returns max activity over channels."""
    C = complex_window.shape[0]
    best = 0.0
    for c in range(C):
        mag = np.sqrt(complex_window[c, ..., 0] ** 2 + complex_window[c, ..., 1] ** 2)
        mag = np.log1p(mag).astype(np.float32)
        mag = (mag - mean) / std
        x = torch.from_numpy(mag).float().unsqueeze(0).unsqueeze(0).to(device)
        x = _pad_to_multiple(x, 32)
        out = model(x)[0]  # (1,2,H,W)
        sig = torch.sigmoid(out[0]).cpu().numpy()  # (2,H,W)
        best = max(best, activity_score_from_sigmoid(sig, threshold))
    return best


def compute_logmag_stats(h5_path, max_samples: int = 60) -> tuple[float, float]:
    """Per-modality log-magnitude mean/std from a sample of step_2a windows.

    Each diagnostic has its own intensity scale and 1/f structure, so the filter
    model must see inputs normalized with that modality's own statistics (the
    surrogate was trained with per-modality normalization). Computed
    automatically here rather than hard-coded per diagnostic.
    """
    from .utils.hdf5_io import get_sample_count, read_sample

    n = get_sample_count(h5_path)
    if n == 0:
        return 0.0, 1.0
    step = max(1, n // max_samples)
    vals = []
    for idx in range(0, n, step):
        data = read_sample(h5_path, idx)  # (C, F, T, 2)
        mag = np.log1p(np.sqrt(data[..., 0] ** 2 + data[..., 1] ** 2))
        vals.append(mag.reshape(-1))
    allv = np.concatenate(vals)
    mean = float(allv.mean())
    std = float(allv.std())
    return mean, (std if std > 1e-6 else 1.0)


def select_window_indices(
    scores: dict[int, float], max_windows: int, min_activity: float
) -> list[int]:
    """Keep indices above floor, top-`max_windows` by score, sorted by score desc."""
    above = [(i, s) for i, s in scores.items() if s >= min_activity]
    above.sort(key=lambda t: t[1], reverse=True)
    return [i for i, _ in above[:max_windows]]
