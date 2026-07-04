from __future__ import annotations

import numpy as np

from tokeye.transforms import compute_stft


def _sine_signal(n_samples: int = 4096, freq: float = 0.05) -> np.ndarray:
    t = np.arange(n_samples)
    return np.sin(2 * np.pi * freq * t).astype(np.float64)[np.newaxis, :]


def test_compute_stft_output_is_2d():
    arr = _sine_signal()
    out = compute_stft(arr, n_fft=256, hop=64)
    assert out.ndim == 2


def test_clip_dc_removes_one_frequency_row():
    arr = _sine_signal()
    with_dc_clip = compute_stft(arr, n_fft=256, hop=64, clip_dc=True)
    without_dc_clip = compute_stft(arr, n_fft=256, hop=64, clip_dc=False)
    assert with_dc_clip.shape[0] == without_dc_clip.shape[0] - 1
    assert with_dc_clip.shape[1] == without_dc_clip.shape[1]


def test_expected_frame_count():
    # For n_samples=4096, n_fft=1024, hop=256, scipy's ShortTimeFFT (with its
    # edge-padding convention) yields exactly 19 frames; pin that number so a
    # change to the padding/framing convention is caught as a regression.
    n_samples = 4096
    n_fft = 1024
    hop = 256
    arr = _sine_signal(n_samples=n_samples)
    out = compute_stft(arr, n_fft=n_fft, hop=hop)
    assert out.shape[1] == 19


def test_values_within_percentile_clip_bounds():
    # A signal with an extreme outlier burst has a much wider raw dynamic
    # range than its 1st-99th percentile band; percentile clipping should
    # compress the output range to (approximately) that narrower band.
    rng = np.random.default_rng(0)
    arr = rng.normal(size=4096)
    arr[2000] = 1000.0  # single extreme outlier
    arr = arr[np.newaxis, :].astype(np.float64)

    clip_low, clip_high = 1.0, 99.0
    clipped = compute_stft(
        arr, n_fft=256, hop=64, clip_low=clip_low, clip_high=clip_high
    )
    unclipped = compute_stft(arr, n_fft=256, hop=64, clip_low=0.0, clip_high=100.0)

    assert np.all(np.isfinite(clipped))
    assert clipped.min() >= unclipped.min()
    assert clipped.max() <= unclipped.max()
    assert clipped.max() < unclipped.max()  # outlier should be visibly clipped
