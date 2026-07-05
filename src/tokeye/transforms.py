from __future__ import annotations

import numpy as np
from scipy import signal

DEFAULT_N_FFT = 1024
DEFAULT_HOP = 256  # app UI default; training recipe used 128
DEFAULT_CLIP_DC = True
DEFAULT_CLIP_LOW = 1.0
DEFAULT_CLIP_HIGH = 99.0


def compute_stft(
    arr: np.ndarray,
    n_fft: int = DEFAULT_N_FFT,
    hop: int = DEFAULT_HOP,
    window: str = "hann",
    clip_dc: bool = DEFAULT_CLIP_DC,
    fs: float = 1.0,
    clip_low: float = DEFAULT_CLIP_LOW,
    clip_high: float = DEFAULT_CLIP_HIGH,
) -> np.ndarray:
    win = signal.get_window(window, n_fft)
    transform = signal.ShortTimeFFT(win=win, hop=hop, fs=fs)
    sxx = transform.stft(arr)

    if sxx.shape[0] == 2:
        sxx = sxx[0] * np.conj(sxx[1])
    elif sxx.shape[0] == 1:
        sxx = sxx[0]

    sxx = np.abs(sxx)
    sxx = np.log1p(sxx)

    # DC clipping
    if clip_dc:
        sxx = sxx[1:, :]

    # Percentile clipping
    vmin, vmax = np.percentile(
        sxx,
        [clip_low, clip_high],
    )
    return np.clip(sxx, vmin, vmax)
