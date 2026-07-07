"""Automatic parameter derivation for the single-scale pipeline.

All functions are computed **once per run** from a sample of the data,
producing stable values that are recorded in ``resolved_params.yaml`` and
reused across all samples in that run.

Ported from ``big_tf_unet_ablation`` and extended with:

- robust (median/MAD) statistics and the ``N_a`` asinh normalization used at
  every model-facing site,
- an energy-level edge-bin detector (the gradient detector masks only 1 of
  bes's ~70-bin low-frequency plateau),
- a Kneedle knee-point threshold via the ``kneed`` package (replacing the
  hand-rolled triangle method),
- scale-covariant auto values (baseline ``lam``, UNet ``num_layers``,
  ``batch_size``) so one config works at every (nfft, hop).
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import h5py
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# Reference geometry: nfft=1024/hop=128 with subseq_len=66000 -> (513, 516).
_REF_FREQ_BINS = 513
_REF_TIME_BINS = 516


# ---------------------------------------------------------------------------
# Sampling helper
# ---------------------------------------------------------------------------

def _sample_arrays(h5_path: Path, max_samples: int) -> list[np.ndarray]:
    """Read up to *max_samples* random sample arrays from a step HDF5 file."""
    arrays: list[np.ndarray] = []
    with h5py.File(h5_path, "r") as f:
        grp = f["samples"]
        keys = list(grp.keys())
        rng = np.random.default_rng(42)
        indices = rng.choice(
            len(keys), size=min(max_samples, len(keys)), replace=False
        )
        for idx in indices:
            arrays.append(np.asarray(grp[keys[idx]]))
    return arrays


# ---------------------------------------------------------------------------
# Robust statistics + N_a normalization
# ---------------------------------------------------------------------------

def robust_stats(
    x: np.ndarray, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Median and scaled MAD (1.4826*MAD ~ sigma for Gaussian data).

    Median/MAD have a 50% breakdown point, so a strong mode cannot inflate the
    scale and suppress weak modes the way per-sample mean/std does.
    """
    med = np.median(x, axis=axis, keepdims=True)
    mad = np.median(np.abs(x - med), axis=axis, keepdims=True)
    scale = 1.4826 * mad + 1e-6
    if not keepdims:
        med = np.squeeze(med, axis=axis) if axis is not None else med.squeeze()
        scale = np.squeeze(scale, axis=axis) if axis is not None else scale.squeeze()
    return med, scale


def normalize_asinh(
    x: np.ndarray,
    a: float,
    median: np.ndarray | float,
    scale: np.ndarray | float,
) -> np.ndarray:
    """``N_a(x) = a * asinh((x - median) / (a * scale))``.

    Invertible smooth compression: unit slope through the bulk (|z| << a it IS
    the robust z-score), logarithmic in the tails. ``a`` sets where compression
    starts — a=3 is the smooth generalization of the old hard clip(z, -3, 3);
    unlike clip (or tanh, which saturates), extremes stay ordered and
    gradients never die.
    """
    return a * np.arcsinh((x - median) / (a * scale))


# ---------------------------------------------------------------------------
# Automatic clamp_range  (legacy zscore path of the denoiser)
# ---------------------------------------------------------------------------

def compute_clamp_range(
    h5_path: Path,
    percentiles: tuple[float, float] = (1.0, 99.0),
    max_samples: int = 100,
) -> tuple[float, float]:
    """Global percentile clamp, used only by the legacy zscore normalization."""
    values = [a.ravel() for a in _sample_arrays(h5_path, max_samples)]
    all_vals = np.concatenate(values)
    lo, hi = np.percentile(all_vals, percentiles)
    logger.info(f"Auto clamp_range: ({lo:.4f}, {hi:.4f}) from {len(values)} samples")
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Edge-bin detection  (step_2, step_3)
# ---------------------------------------------------------------------------

def _energy_profile(arrays: list[np.ndarray]) -> np.ndarray:
    """Median per-frequency-bin energy profile across sampled windows."""
    profiles = []
    for arr in arrays:
        if arr.ndim == 4:  # (C, F, T, Z)
            energy = np.sqrt(np.mean(arr**2, axis=(0, 2, 3)))
        elif arr.ndim == 3:  # (C, F, T)
            energy = np.mean(np.abs(arr), axis=(0, 2))
        else:  # (F, T)
            energy = np.mean(np.abs(arr), axis=-1)
        profiles.append(energy)
    return np.median(np.stack(profiles), axis=0)


def detect_edge_bins(
    h5_path: Path,
    gradient_threshold: float = 0.5,
    max_samples: int = 100,
) -> tuple[int, int]:
    """Legacy gradient-based edge detector (kept for comparison).

    Known weakness: a broad edge plateau (bes: ~70 bins) has a small interior
    gradient, so only ~1 bin gets masked. Prefer ``detect_edge_bins_energy``.
    """
    median_energy = _energy_profile(_sample_arrays(h5_path, max_samples))
    grad = np.gradient(median_energy)
    grad_norm = np.abs(grad) / (np.max(np.abs(grad)) + 1e-8)

    lower_idx = 0
    for i in range(len(grad_norm)):
        if grad_norm[i] < gradient_threshold:
            lower_idx = i
            break
    upper_idx = 0
    for i in range(len(grad_norm) - 1, -1, -1):
        if grad_norm[i] < gradient_threshold:
            upper_idx = len(grad_norm) - 1 - i
            break

    lower_idx = max(1, lower_idx)
    upper_idx = max(1, upper_idx)
    logger.info(f"Auto edge bins (gradient): lower={lower_idx}, upper={upper_idx}")
    return lower_idx, upper_idx


def detect_edge_bins_energy(
    h5_path: Path,
    k: float = 2.0,
    max_fraction: float = 0.15,
    max_samples: int = 100,
) -> tuple[int, int]:
    """Energy-level edge-bin detection.

    From the median per-bin energy profile E(f): the interior band (excluding
    a ``max_fraction`` margin at each end) gives a reference median M; from
    each edge, the contiguous run of bins with ``|log E(f) - log M| > log k``
    is masked, capped at ``max_fraction`` of the bins per edge.

    Index-based on the run's own profile, so it adapts to any (nfft, hop) and
    catches broad plateaus (bes) that the gradient criterion misses.
    """
    profile = _energy_profile(_sample_arrays(h5_path, max_samples))
    n = len(profile)
    margin = max(1, int(n * max_fraction))
    interior = profile[margin : n - margin]
    log_m = np.log(np.median(interior) + 1e-12)
    dev = np.abs(np.log(profile + 1e-12) - log_m)
    log_k = np.log(k)

    cap = max(1, int(n * max_fraction))
    lower_idx = 0
    while lower_idx < cap and dev[lower_idx] > log_k:
        lower_idx += 1
    upper_idx = 0
    while upper_idx < cap and dev[n - 1 - upper_idx] > log_k:
        upper_idx += 1

    lower_idx = max(1, lower_idx)
    upper_idx = max(1, upper_idx)
    logger.info(
        f"Auto edge bins (energy): lower={lower_idx}, upper={upper_idx} "
        f"(k={k}, cap={cap} of {n} bins/edge)"
    )
    return lower_idx, upper_idx


# ---------------------------------------------------------------------------
# Knee-point threshold  (step_4)
# ---------------------------------------------------------------------------

def knee_threshold(
    z: np.ndarray,
    sensitivity: float = 1.0,
    delta: float = 0.0,
    fallback_frac: float = 0.02,
    grid_points: int = 512,
) -> dict:
    """Kneedle threshold on the ECDF of positive robust-z values.

    ``z`` must already be robust-standardized ((x - median) / (1.4826*MAD)),
    NOT asinh-compressed — knee location is not invariant under monotone
    compression. Modes are positive excess by construction, so only ``z > 0``
    enters the curve (equivalent to the old clamp-below-median).

    Returns a dict with ``threshold`` (in robust-sigma units), ``knee``,
    ``used_fallback``, and ``positive_fraction`` (fraction of z above the
    threshold) for logging into thresholds.csv.
    """
    from kneed import KneeLocator

    zp = np.sort(z[z > 0].ravel())
    if zp.size < 100:
        t = float("inf")
        return {
            "threshold": t,
            "knee": None,
            "used_fallback": True,
            "positive_fraction": 0.0,
        }

    x = np.linspace(0.0, float(np.quantile(zp, 0.999)), grid_points)
    y = np.searchsorted(zp, x, side="right") / zp.size  # ECDF (no data clipped)

    knee = None
    try:
        kl = KneeLocator(
            x, y, curve="concave", direction="increasing", S=sensitivity
        )
        knee = float(kl.knee) if kl.knee is not None else None
    except Exception:  # noqa: BLE001 — kneed can fail on degenerate curves
        knee = None

    used_fallback = knee is None
    base = knee if knee is not None else float(np.quantile(zp, 1 - fallback_frac))
    if used_fallback:
        logger.warning(
            f"No knee found; falling back to the {1 - fallback_frac:.3f} quantile"
        )
    t = base + delta
    positive_fraction = float(np.mean(z > t))
    return {
        "threshold": t,
        "knee": knee,
        "used_fallback": used_fallback,
        "positive_fraction": positive_fraction,
    }


# ---------------------------------------------------------------------------
# Scale-covariant autos
# ---------------------------------------------------------------------------

def compute_lam(
    freq_bins: int, base_lam: float = 1.0e6, base_freq_bins: int = _REF_FREQ_BINS
) -> float:
    """Baseline stiffness with constant *physical* smoothing across scales.

    The Whittaker-type smoother's halfwidth scales ~lam^(1/4) in bin units, so
    lam must scale with (freq_bins/base)^4 to smooth the same physical
    frequency span at every nfft.
    """
    lam = base_lam * (freq_bins / base_freq_bins) ** 4
    logger.info(f"Auto lam: {lam:.3e} (freq_bins={freq_bins})")
    return float(lam)


def compute_num_layers(freq_bins: int, time_bins: int, cap: int = 5) -> int:
    """UNet depth that keeps the bottleneck at least ~4 px in each dim."""
    depth = min(cap, int(math.floor(math.log2(min(freq_bins, time_bins) / 4))))
    depth = max(1, depth)
    logger.info(f"Auto num_layers: {depth} (F={freq_bins}, T={time_bins})")
    return depth


def pad_to_multiple(n: int, multiple: int) -> int:
    return int(math.ceil(n / multiple) * multiple)


def compute_batch_size(
    base_batch_size: int, freq_bins: int, time_bins: int, num_layers: int
) -> int:
    """Scale the reference batch size by the per-sample activation footprint."""
    mult = 2**num_layers
    f_pad = pad_to_multiple(freq_bins, mult)
    t_pad = pad_to_multiple(time_bins, mult)
    ref = _REF_FREQ_BINS * _REF_TIME_BINS
    batch = max(1, int(base_batch_size * ref / (f_pad * t_pad)))
    logger.info(f"Auto batch_size: {batch} (pad {f_pad}x{t_pad}, base {base_batch_size})")
    return batch


# ---------------------------------------------------------------------------
# Row removal / min size  (step_4)
# ---------------------------------------------------------------------------

def compute_row_removal(
    freq_bins: int,
    bottom_fraction: float = 0.01,
    top_fraction: float = 0.004,
) -> tuple[int, int]:
    """Rows to strip from the mask edges, as a fraction of frequency bins.

    For nfft=1024 (freq_bins=513): bottom=5, top=2 — the original defaults.
    """
    remove_bottom = max(1, int(freq_bins * bottom_fraction))
    remove_top = max(1, int(freq_bins * top_fraction))
    logger.info(f"Auto row removal: bottom={remove_bottom}, top={remove_top}")
    return remove_bottom, remove_top


def compute_min_size(height: int, width: int, fraction: float = 0.0002) -> int:
    """Minimum connected-component size as a fraction of image area."""
    min_size = max(1, int(height * width * fraction))
    logger.info(f"Auto min_size: {min_size} (image {height}x{width})")
    return min_size


# ---------------------------------------------------------------------------
# Per-modality robust log-magnitude stats  (step_1 window filter, step_5)
# ---------------------------------------------------------------------------

def compute_logmag_robust_stats(
    h5_path: Path, max_samples: int = 100
) -> tuple[float, float]:
    """Per-modality median/scale of log1p(|Z|) from sampled spectrograms.

    Each diagnostic has its own intensity scale and 1/f structure, so these
    stats must come from that modality's own data — never a global constant.
    """
    values = []
    for arr in _sample_arrays(h5_path, max_samples):
        if arr.ndim == 4:  # (C, F, T, 2) real/imag
            mag = np.sqrt(arr[..., 0] ** 2 + arr[..., 1] ** 2)
        else:
            mag = np.abs(arr)
        values.append(np.log1p(mag).ravel())
    all_vals = np.concatenate(values)
    med, scale = robust_stats(all_vals)
    logger.info(f"Auto logmag stats: median={med:.4f}, scale={scale:.4f}")
    return float(med), float(scale)
