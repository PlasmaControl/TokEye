"""
Signal Processing Transformations

This module provides core signal processing transformations for TokEye,
including preemphasis filtering, STFT computation, and wavelet decomposition.
"""

import warnings
from typing import Literal, Optional

import numpy as np
import pywt
from scipy import signal as scipy_signal

from TokEye.exceptions import InvalidSignalError, TransformError


def apply_preemphasis(signal: np.ndarray, alpha: float = 0.97) -> np.ndarray:
    """
    Apply preemphasis filter to enhance high-frequency components.

    Implements the difference equation: y[n] = x[n] - α * x[n-1]

    This filter emphasizes higher frequencies in the signal, which can improve
    the discriminability of spectral features in plasma signals.

    Args:
        signal: Input signal array, 1D or 2D (channels, samples)
        alpha: Preemphasis coefficient, typically 0.95-0.97

    Returns:
        Preemphasized signal with same shape as input

    Raises:
        ValueError: If alpha is not in valid range [0, 1]
        ValueError: If signal is not 1D or 2D

    Example:
        >>> signal = np.random.randn(1000)
        >>> emphasized = apply_preemphasis(signal, alpha=0.97)
    """
    if not 0 <= alpha <= 1:
        raise InvalidSignalError(f"Alpha must be in range [0, 1], got {alpha}")

    if signal.ndim == 0 or signal.ndim > 2:
        raise InvalidSignalError(f"Signal must be 1D or 2D, got {signal.ndim}D")

    # Handle empty signal
    if signal.size == 0:
        return signal.copy()

    # Apply preemphasis filter
    if signal.ndim == 1:
        # 1D signal: y[n] = x[n] - alpha * x[n-1]
        output = np.empty_like(signal)
        output[0] = signal[0]
        output[1:] = signal[1:] - alpha * signal[:-1]
    else:
        # 2D signal: apply to each channel independently
        output = np.empty_like(signal)
        output[:, 0] = signal[:, 0]
        output[:, 1:] = signal[:, 1:] - alpha * signal[:, :-1]

    return output


def compute_stft(
    signal: np.ndarray,
    n_fft: int = 1024,
    hop_length: int = 128,
    window: str = "hann",
    clip_dc: bool = True,
    fs: float = 1.0,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
) -> np.ndarray:
    """
    Compute Short-Time Fourier Transform with normalization.

    This function computes the STFT and applies the following transformations:
    1. Magnitude spectrum
    2. Log compression: log1p(magnitude)
    3. Optional DC component clipping
    4. Percentile clipping
    5. Mean-std normalization

    Args:
        signal: Input time-domain signal (1D array)
        n_fft: FFT size (number of frequency bins)
        hop_length: Number of samples between successive frames
        window: Window function name ('hann', 'hamming', 'blackman', etc.)
        clip_dc: If True, remove DC component (first frequency bin)
        fs: Sampling frequency in Hz
        percentile_low: Lower percentile for clipping (default: 1.0)
        percentile_high: Upper percentile for clipping (default: 99.0)

    Returns:
        Normalized STFT magnitude spectrogram (freq_bins, time_frames)
        If clip_dc=True, shape is (freq_bins-1, time_frames)

    Raises:
        ValueError: If signal is not 1D
        ValueError: If n_fft or hop_length are invalid

    Example:
        >>> signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))
        >>> spectrogram = compute_stft(signal, n_fft=2048, hop_length=512)
    """
    if signal.ndim != 1:
        raise InvalidSignalError(f"Signal must be 1D, got {signal.ndim}D")

    if n_fft <= 0:
        raise InvalidSignalError(f"n_fft must be positive, got {n_fft}")

    if hop_length <= 0:
        raise InvalidSignalError(f"hop_length must be positive, got {hop_length}")

    if signal.size == 0:
        raise InvalidSignalError("Cannot compute STFT of empty signal")

    # Compute STFT
    try:
        frequencies, times, stft_matrix = scipy_signal.stft(
            signal,
            fs=fs,
            window=window,
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
            nfft=n_fft,
            return_onesided=True,
            boundary="zeros",
            padded=True,
        )
    except Exception as e:
        raise TransformError(f"STFT computation failed: {e}")

    # Compute magnitude spectrum
    magnitude = np.abs(stft_matrix)

    # Log compression (log1p for numerical stability)
    result = np.log1p(magnitude)

    # Optional DC clipping
    if clip_dc:
        result = result[1:, :]  # Remove first frequency bin

    # Percentile clipping
    if percentile_low > 0 or percentile_high < 100:
        low_val = np.percentile(result, percentile_low)
        high_val = np.percentile(result, percentile_high)
        result = np.clip(result, low_val, high_val)

    # Mean-std normalization
    mean = np.mean(result)
    std = np.std(result)

    if std < 1e-10:
        warnings.warn(
            "Standard deviation is near zero, normalization may be unstable",
            RuntimeWarning,
        )
        normalized = result - mean
    else:
        normalized = (result - mean) / (std + 1e-8)

    return normalized


def compute_wavelet(
    signal: np.ndarray,
    wavelet: str = "db8",
    level: int = 9,
    mode: str = "symmetric",
    order: Literal["natural", "freq"] = "freq",
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
) -> np.ndarray:
    """
    Compute wavelet packet decomposition as specified in TokEye requirements.

    This function performs a strict wavelet packet transform implementation:
    1. Create WaveletPacket with specified parameters
    2. Extract nodes at target decomposition level
    3. Apply log1p transformation to absolute values
    4. Percentile clipping
    5. Mean-std normalization

    Args:
        signal: Input time-domain signal (1D array)
        wavelet: Wavelet name (e.g., 'db8', 'db4', 'haar', 'sym8')
        level: Decomposition level (depth of wavelet tree)
        mode: Signal extension mode ('symmetric', 'zero', 'constant', 'periodic', etc.)
        order: Node ordering ('natural' for tree traversal, 'freq' for frequency order)
        percentile_low: Lower percentile for clipping (default: 1.0)
        percentile_high: Upper percentile for clipping (default: 99.0)

    Returns:
        2D array of wavelet coefficients (num_nodes, coefficients_per_node)
        where num_nodes = 2^level

    Raises:
        ValueError: If signal is not 1D
        ValueError: If level is negative
        ValueError: If wavelet is not supported

    Example:
        >>> signal = np.random.randn(10000)
        >>> coeffs = compute_wavelet(signal, wavelet='db8', level=9)
        >>> print(coeffs.shape)  # (512, N) where N depends on signal length

    Notes:
        - The output is log-compressed for better dynamic range
        - Frequency ordering is recommended for spectral analysis
        - Higher levels produce more nodes but fewer coefficients per node
    """
    if signal.ndim != 1:
        raise InvalidSignalError(f"Signal must be 1D, got {signal.ndim}D")

    if level < 0:
        raise InvalidSignalError(f"Level must be non-negative, got {level}")

    if signal.size == 0:
        raise InvalidSignalError("Cannot compute wavelet transform of empty signal")

    # Verify wavelet is valid
    if wavelet not in pywt.wavelist():
        raise InvalidSignalError(
            f"Wavelet '{wavelet}' not supported. Available wavelets: {pywt.wavelist()}"
        )

    # Verify mode is valid
    valid_modes = pywt.Modes.modes
    if mode not in valid_modes:
        raise InvalidSignalError(
            f"Mode '{mode}' not supported. Available modes: {valid_modes}"
        )

    # Strict implementation as specified
    try:
        wp = pywt.WaveletPacket(signal, wavelet, mode, maxlevel=level)
    except Exception as e:
        raise TransformError(f"WaveletPacket creation failed: {e}")

    # Get nodes at specified level
    try:
        nodes = wp.get_level(level, order=order)
    except Exception as e:
        raise TransformError(f"Failed to extract level {level} nodes: {e}")

    if not nodes:
        raise TransformError(f"No nodes found at level {level}")

    # Extract labels and values as specified
    labels = [n.path for n in nodes]
    result = np.array([n.data for n in nodes], dtype="d")

    # Apply log transformation to absolute values
    result = np.log1p(np.abs(result))

    # Percentile clipping
    if percentile_low > 0 or percentile_high < 100:
        low_val = np.percentile(result, percentile_low)
        high_val = np.percentile(result, percentile_high)
        result = np.clip(result, low_val, high_val)

    # Mean-std normalization
    mean = np.mean(result)
    std = np.std(result)
    result = (result - mean) / (std + 1e-8)

    return result


def compute_wavelet_energy_spectrum(
    signal: np.ndarray,
    wavelet: str = "db8",
    level: int = 9,
    mode: str = "symmetric",
) -> np.ndarray:
    """
    Compute energy spectrum from wavelet decomposition.

    This is a convenience function that computes the wavelet transform
    and returns the energy (squared coefficients sum) per node.

    Args:
        signal: Input time-domain signal (1D array)
        wavelet: Wavelet name
        level: Decomposition level
        mode: Signal extension mode

    Returns:
        1D array of energy values per node (length = 2^level)

    Example:
        >>> signal = np.random.randn(10000)
        >>> energy = compute_wavelet_energy_spectrum(signal)
        >>> print(energy.shape)  # (512,)
    """
    coeffs = compute_wavelet(signal, wavelet, level, mode, order="freq")

    # Compute energy per node (sum of squared coefficients)
    energy = np.sum(coeffs**2, axis=1)

    return energy
