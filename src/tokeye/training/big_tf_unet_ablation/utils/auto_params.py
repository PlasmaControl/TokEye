"""Automatic parameter derivation for the multiscale pipeline.

All functions are computed **once per combo** from a sample of the data,
producing stable values that are reused across all samples in that combo.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import h5py
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Automatic clamp_range  (step 3a)
# ---------------------------------------------------------------------------

def compute_clamp_range(
    h5_path: Path,
    percentiles: tuple[float, float] = (1.0, 99.0),
    max_samples: int = 100,
) -> tuple[float, float]:
    """Compute clamp range from data quantiles.

    Reads up to *max_samples* random spectrograms from the step_2b HDF5
    file and returns the *percentiles* across all values.
    """
    values: list[np.ndarray] = []
    with h5py.File(h5_path, "r") as f:
        grp = f["samples"]
        keys = list(grp.keys())
        rng = np.random.default_rng(42)
        indices = rng.choice(len(keys), size=min(max_samples, len(keys)), replace=False)
        for idx in indices:
            arr = np.asarray(grp[keys[idx]])
            values.append(arr.ravel())

    all_vals = np.concatenate(values)
    lo, hi = np.percentile(all_vals, percentiles)
    logger.info(f"Auto clamp_range: ({lo:.4f}, {hi:.4f}) from {len(indices)} samples")
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# 2. Automatic bin cutting / edge detection  (step 2b, step 3a)
# ---------------------------------------------------------------------------

def detect_edge_bins(
    h5_path: Path,
    gradient_threshold: float = 0.5,
    max_samples: int = 100,
) -> tuple[int, int]:
    """Detect frequency bins at edges where spectral energy drops sharply.

    Computes the mean spectral energy per frequency bin from a sample of
    spectrograms, then finds where the gradient magnitude stabilises.

    Returns:
        ``(lower_idx, upper_idx)`` — number of bins to mask at the bottom
        and top of the spectrogram.
    """
    energy_profiles: list[np.ndarray] = []
    with h5py.File(h5_path, "r") as f:
        grp = f["samples"]
        keys = list(grp.keys())
        rng = np.random.default_rng(42)
        indices = rng.choice(len(keys), size=min(max_samples, len(keys)), replace=False)
        for idx in indices:
            arr = np.asarray(grp[keys[idx]])  # (C, F, T, Z) or (C, F, T)
            # Mean energy per frequency bin across channels, time, and complex dims
            if arr.ndim == 4:
                energy = np.sqrt(np.mean(arr ** 2, axis=(0, 2, 3)))
            elif arr.ndim == 3:
                energy = np.mean(np.abs(arr), axis=(0, 2))
            else:
                energy = np.mean(np.abs(arr), axis=0)
            energy_profiles.append(energy)

    median_energy = np.median(np.stack(energy_profiles), axis=0)
    grad = np.gradient(median_energy)
    grad_norm = np.abs(grad) / (np.max(np.abs(grad)) + 1e-8)

    # Scan from bottom: first bin where gradient drops below threshold
    lower_idx = 0
    for i in range(len(grad_norm)):
        if grad_norm[i] < gradient_threshold:
            lower_idx = i
            break

    # Scan from top
    upper_idx = 0
    for i in range(len(grad_norm) - 1, -1, -1):
        if grad_norm[i] < gradient_threshold:
            upper_idx = len(grad_norm) - 1 - i
            break

    lower_idx = max(1, lower_idx)
    upper_idx = max(1, upper_idx)
    logger.info(
        f"Auto edge bins: lower={lower_idx}, upper={upper_idx} "
        f"(from {len(indices)} samples, threshold={gradient_threshold})"
    )
    return lower_idx, upper_idx


# ---------------------------------------------------------------------------
# 3. Automatic row removal  (step 4a)
# ---------------------------------------------------------------------------

def compute_row_removal(
    freq_bins: int,
    bottom_fraction: float = 0.01,
    top_fraction: float = 0.004,
) -> tuple[int, int]:
    """Compute number of rows to remove from spectrogram edges.

    Based on a fraction of the total frequency bins.  For nfft=1024
    (freq_bins=513): bottom=5, top=2 — matching the original defaults.
    """
    remove_bottom = max(1, int(freq_bins * bottom_fraction))
    remove_top = max(1, int(freq_bins * top_fraction))
    logger.info(
        f"Auto row removal: bottom={remove_bottom}, top={remove_top} "
        f"(freq_bins={freq_bins})"
    )
    return remove_bottom, remove_top


# ---------------------------------------------------------------------------
# 4. Automatic min_size  (step 4a)
# ---------------------------------------------------------------------------

def compute_min_size(
    height: int,
    width: int,
    fraction: float = 0.0002,
) -> int:
    """Compute minimum object size as a fraction of image area.

    Default fraction ≈ 0.0002 gives ~53 for a 513×516 image, close to
    the original hardcoded value of 64.
    """
    min_size = max(1, int(height * width * fraction))
    logger.info(
        f"Auto min_size: {min_size} (image {height}×{width}, fraction={fraction})"
    )
    return min_size
