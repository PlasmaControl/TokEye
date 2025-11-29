import numpy as np
from scipy import signal


def compute_stft(
    arr: np.ndarray,
    n_fft: int = 1024,
    hop: int = 128,
    window: str = "hann",
    clip_dc: bool = True,
    fs: float = 1.0,
    clip_low: float = 1.0,
    clip_high: float = 99.0,
) -> np.ndarray:
    win = signal.get_window(window, n_fft)
    transform = signal.ShortTimeFFT(win=win, hop=hop, fs=fs)
    sxx = transform.stft(arr)

    if sxx.shape[0] == 2:
        sxx = sxx[0] * np.conj(sxx[1])
    elif sxx.shape[0] == 1:
        sxx = sxx[0]
    else:
        raise ValueError("Sxx can only have 1 or 2 channels")

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
    sxx = np.clip(sxx, vmin, vmax)

    return sxx
