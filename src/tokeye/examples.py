"""Deterministic synthetic example signal for demos and smoke tests.

Produces a 1D signal with Gaussian noise plus a handful of sustained
frequency-ramp "chirps" (coherent activity) and short Gaussian-windowed
"bursts" (transient activity), so a log1p spectrogram of the output shows
both activity classes TokEye is trained to segment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import chirp
from scipy.signal.windows import tukey

NOISE_SIGMA = 1.0
CHIRP_AMPLITUDE = 3.0
BURST_AMPLITUDE = 5.0


def _add_chirp(
    t: np.ndarray, fs: float, duration_s: float, rng: np.random.Generator
) -> np.ndarray:
    """A sustained linear/quadratic frequency ramp over part of the signal."""
    seg_duration = rng.uniform(0.3, 0.6) * duration_s
    start_time = rng.uniform(0.0, duration_s - seg_duration)
    mask = (t >= start_time) & (t < start_time + seg_duration)

    local_t = t[mask] - start_time
    f0 = rng.uniform(0.02, 0.10) * fs
    f1 = rng.uniform(0.10, 0.30) * fs
    method = rng.choice(["linear", "quadratic"])
    ramp = chirp(local_t, f0=f0, f1=f1, t1=seg_duration, method=method)
    envelope = tukey(local_t.size, alpha=0.1) if local_t.size else local_t

    out = np.zeros_like(t)
    out[mask] = CHIRP_AMPLITUDE * ramp * envelope
    return out


def _add_burst(
    t: np.ndarray, fs: float, duration_s: float, rng: np.random.Generator
) -> np.ndarray:
    """A short Gaussian-windowed tone burst (transient activity)."""
    burst_duration = rng.uniform(0.002, 0.01) * duration_s
    center_time = rng.uniform(0.0, duration_s)
    carrier_freq = rng.uniform(0.05, 0.40) * fs
    phase = rng.uniform(0.0, 2 * np.pi)

    sigma = burst_duration / 6.0
    envelope = np.exp(-0.5 * ((t - center_time) / sigma) ** 2)
    return BURST_AMPLITUDE * envelope * np.sin(2 * np.pi * carrier_freq * t + phase)


def make_example_signal(
    duration_s: float = 2.0,
    fs: float = 200_000.0,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_samples = int(duration_s * fs)
    t = np.arange(n_samples, dtype=np.float64) / fs

    sig = rng.normal(0.0, NOISE_SIGMA, size=n_samples)

    for _ in range(rng.integers(2, 4)):  # 2-3 chirps
        sig += _add_chirp(t, fs, duration_s, rng)

    for _ in range(rng.integers(2, 4)):  # 2-3 bursts
        sig += _add_burst(t, fs, duration_s, rng)

    return sig.astype(np.float32)


def write_example_signal(
    directory: Path,
    filename: str = "tokeye_example.npy",
) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    out_path = directory / filename
    np.save(out_path, make_example_signal())
    return out_path
