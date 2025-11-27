"""
Tests for signal processing transformations.

This module tests all functions in TokEye.processing.transforms, including:
- Preemphasis filtering
- STFT computation
- Wavelet decomposition
"""

import warnings

import numpy as np
import pytest

from TokEye.exceptions import InvalidSignalError, TransformError
from TokEye.processing.transforms import (
    apply_preemphasis,
    compute_stft,
    compute_wavelet,
    compute_wavelet_energy_spectrum,
)


class TestApplyPreemphasis:
    """Test suite for apply_preemphasis function."""

    def test_1d_signal_basic(self, sample_signal_1d):
        """Test preemphasis on 1D signal."""
        result = apply_preemphasis(sample_signal_1d, alpha=0.97)

        assert result.shape == sample_signal_1d.shape
        assert result.dtype == sample_signal_1d.dtype
        assert result[0] == sample_signal_1d[0]  # First sample unchanged

    def test_2d_signal_basic(self, sample_signal_2d):
        """Test preemphasis on 2D multi-channel signal."""
        result = apply_preemphasis(sample_signal_2d, alpha=0.97)

        assert result.shape == sample_signal_2d.shape
        assert result.dtype == sample_signal_2d.dtype
        assert np.allclose(result[:, 0], sample_signal_2d[:, 0])

    def test_alpha_zero(self, sample_signal_1d):
        """Test with alpha=0 (no preemphasis, should return copy)."""
        result = apply_preemphasis(sample_signal_1d, alpha=0.0)
        assert np.allclose(result, sample_signal_1d)

    def test_alpha_one(self, sample_signal_1d):
        """Test with alpha=1 (full differentiation)."""
        result = apply_preemphasis(sample_signal_1d, alpha=1.0)

        # With alpha=1: result[n] = signal[n] - signal[n-1]
        expected_diff = sample_signal_1d[1:] - sample_signal_1d[:-1]
        assert np.allclose(result[1:], expected_diff)

    def test_invalid_alpha_negative(self, sample_signal_1d):
        """Test that negative alpha raises ValueError."""
        with pytest.raises(InvalidSignalError, match="Alpha must be in range"):
            apply_preemphasis(sample_signal_1d, alpha=-0.1)

    def test_invalid_alpha_greater_than_one(self, sample_signal_1d):
        """Test that alpha > 1 raises ValueError."""
        with pytest.raises(InvalidSignalError, match="Alpha must be in range"):
            apply_preemphasis(sample_signal_1d, alpha=1.5)

    def test_invalid_dimensionality(self):
        """Test that 0D or >2D signals raise ValueError."""
        # 0D signal (scalar)
        with pytest.raises(InvalidSignalError, match="Signal must be 1D or 2D"):
            apply_preemphasis(np.array(5.0), alpha=0.97)

        # 3D signal
        signal_3d = np.random.randn(3, 3, 100)
        with pytest.raises(InvalidSignalError, match="Signal must be 1D or 2D"):
            apply_preemphasis(signal_3d, alpha=0.97)

    def test_empty_signal(self):
        """Test that empty signal is handled correctly."""
        empty = np.array([])
        result = apply_preemphasis(empty, alpha=0.97)
        assert result.shape == empty.shape
        assert result.size == 0

    def test_single_sample(self):
        """Test signal with single sample."""
        signal = np.array([1.5])
        result = apply_preemphasis(signal, alpha=0.97)
        assert result.shape == (1,)
        assert result[0] == signal[0]

    def test_default_alpha(self, sample_signal_1d):
        """Test that default alpha=0.97 is applied correctly."""
        result = apply_preemphasis(sample_signal_1d)
        assert result.shape == sample_signal_1d.shape


class TestComputeSTFT:
    """Test suite for compute_stft function."""

    def test_basic_computation(self, sample_signal_1d):
        """Test basic STFT computation."""
        result = compute_stft(sample_signal_1d, n_fft=256, hop_length=64)

        assert result.ndim == 2
        assert result.shape[0] > 0  # Has frequency bins
        assert result.shape[1] > 0  # Has time frames

    def test_dc_clipping(self, sample_signal_1d):
        """Test that DC component is clipped when clip_dc=True."""
        with_dc = compute_stft(sample_signal_1d, n_fft=256, clip_dc=False)
        without_dc = compute_stft(sample_signal_1d, n_fft=256, clip_dc=True)

        # With DC should have one more frequency bin
        assert with_dc.shape[0] == without_dc.shape[0] + 1

    def test_window_types(self, sample_signal_1d):
        windows = ["hann", "hamming", "blackman"]

        for window in windows:
            result = compute_stft(sample_signal_1d, window=window)
            assert result.ndim == 2

    def test_different_fft_sizes(self, sample_signal_1d):
        """Test various FFT sizes."""
        fft_sizes = [128, 256, 512, 1024]

        for n_fft in fft_sizes:
            result = compute_stft(sample_signal_1d, n_fft=n_fft, clip_dc=False)
            # Frequency bins = n_fft // 2 + 1 (one-sided)
            assert result.shape[0] == n_fft // 2 + 1

    def test_hop_length_effect(self, sample_signal_1d):
        """Test that smaller hop_length produces more time frames."""
        result_large_hop = compute_stft(sample_signal_1d, hop_length=256)
        result_small_hop = compute_stft(sample_signal_1d, hop_length=64)

        assert result_small_hop.shape[1] > result_large_hop.shape[1]

    def test_invalid_signal_dimension(self):
        """Test that non-1D signal raises ValueError."""
        signal_2d = np.random.randn(3, 1000)
        with pytest.raises(InvalidSignalError, match="Signal must be 1D"):
            compute_stft(signal_2d)

    def test_invalid_n_fft(self, sample_signal_1d):
        """Test that invalid n_fft raises ValueError."""
        with pytest.raises(InvalidSignalError, match="n_fft must be positive"):
            compute_stft(sample_signal_1d, n_fft=0)

        with pytest.raises(InvalidSignalError, match="n_fft must be positive"):
            compute_stft(sample_signal_1d, n_fft=-128)

    def test_invalid_hop_length(self, sample_signal_1d):
        """Test that invalid hop_length raises ValueError."""
        with pytest.raises(InvalidSignalError, match="hop_length must be positive"):
            compute_stft(sample_signal_1d, hop_length=0)

        with pytest.raises(InvalidSignalError, match="hop_length must be positive"):
            compute_stft(sample_signal_1d, hop_length=-64)

    def test_empty_signal(self):
        """Test that empty signal raises ValueError."""
        empty = np.array([])
        with pytest.raises(
            InvalidSignalError, match="Cannot compute STFT of empty signal"
        ):
            compute_stft(empty)

    def test_percentile_clipping(self, sample_signal_1d):
        """Test percentile clipping functionality."""
        result = compute_stft(
            sample_signal_1d, percentile_low=5.0, percentile_high=95.0
        )
        assert result.ndim == 2

    def test_normalization(self, sample_signal_1d):
        """Test that output is normalized (mean~0, std~1)."""
        result = compute_stft(sample_signal_1d)

        # Should be approximately normalized
        assert abs(np.mean(result)) < 0.5
        assert 0.5 < np.std(result) < 2.0


class TestComputeWavelet:
    """Test suite for compute_wavelet function."""

    def test_basic_computation(self, sample_signal_1d):
        """Test basic wavelet decomposition."""
        result = compute_wavelet(sample_signal_1d, wavelet="db8", level=5)

        assert result.ndim == 2
        assert result.shape[0] == 2**5  # Number of nodes = 2^level

    def test_different_levels(self, sample_signal_1d):
        """Test various decomposition levels."""
        levels = [3, 5, 7]

        for level in levels:
            result = compute_wavelet(sample_signal_1d, level=level)
            assert result.shape[0] == 2**level

    def test_different_wavelets(self, sample_signal_1d):
        """Test different wavelet families."""
        wavelets = ["db4", "db8", "sym8", "haar"]

        for wavelet in wavelets:
            result = compute_wavelet(sample_signal_1d, wavelet=wavelet, level=4)
            assert result.ndim == 2
            assert result.shape[0] == 2**4

    def test_frequency_ordering(self, sample_signal_1d):
        """Test frequency ordering of nodes."""
        result_freq = compute_wavelet(sample_signal_1d, order="freq", level=4)
        result_natural = compute_wavelet(sample_signal_1d, order="natural", level=4)

        # Both should have same shape
        assert result_freq.shape == result_natural.shape
        # But different values due to ordering
        assert not np.allclose(result_freq, result_natural)

    def test_different_modes(self, sample_signal_1d):
        """Test different signal extension modes."""
        modes = ["symmetric", "zero", "constant", "periodic"]

        for mode in modes:
            result = compute_wavelet(sample_signal_1d, mode=mode, level=4)
            assert result.ndim == 2

    def test_invalid_signal_dimension(self):
        """Test that non-1D signal raises ValueError."""
        signal_2d = np.random.randn(3, 1000)
        with pytest.raises(InvalidSignalError, match="Signal must be 1D"):
            compute_wavelet(signal_2d)

    def test_invalid_level(self, sample_signal_1d):
        """Test that negative level raises ValueError."""
        with pytest.raises(InvalidSignalError, match="Level must be non-negative"):
            compute_wavelet(sample_signal_1d, level=-1)

    def test_invalid_wavelet(self, sample_signal_1d):
        """Test that unsupported wavelet raises ValueError."""
        with pytest.raises(InvalidSignalError, match="Wavelet .* not supported"):
            compute_wavelet(sample_signal_1d, wavelet="invalid_wavelet")

    def test_invalid_mode(self, sample_signal_1d):
        """Test that unsupported mode raises ValueError."""
        with pytest.raises(InvalidSignalError, match="Mode .* not supported"):
            compute_wavelet(sample_signal_1d, mode="invalid_mode")

    def test_empty_signal(self):
        """Test that empty signal raises ValueError."""
        empty = np.array([])
        with pytest.raises(
            InvalidSignalError, match="Cannot compute wavelet transform"
        ):
            compute_wavelet(empty)

    def test_percentile_clipping(self, sample_signal_1d):
        """Test percentile clipping in wavelet transform."""
        result = compute_wavelet(
            sample_signal_1d, percentile_low=5.0, percentile_high=95.0, level=4
        )
        assert result.ndim == 2

    def test_normalization(self, sample_signal_1d):
        """Test that output is normalized."""
        result = compute_wavelet(sample_signal_1d, level=5)

        # Should be approximately normalized
        assert abs(np.mean(result)) < 0.5
        assert 0.5 < np.std(result) < 2.0

    def test_zero_level(self, sample_signal_1d):
        """Test level=0 decomposition raises RuntimeError."""
        # Level 0 doesn't produce nodes in pywavelets
        with pytest.raises(TransformError, match="No nodes found"):
            compute_wavelet(sample_signal_1d, level=0)


class TestComputeWaveletEnergySpectrum:
    """Test suite for compute_wavelet_energy_spectrum function."""

    def test_basic_computation(self, sample_signal_1d):
        """Test basic energy spectrum computation."""
        result = compute_wavelet_energy_spectrum(sample_signal_1d, level=5)

        assert result.ndim == 1
        assert result.shape[0] == 2**5

    def test_energy_non_negative(self, sample_signal_1d):
        """Test that energy values are non-negative."""
        result = compute_wavelet_energy_spectrum(sample_signal_1d, level=4)
        assert np.all(result >= 0)

    def test_different_levels(self, sample_signal_1d):
        """Test energy spectrum at different levels."""
        levels = [3, 5, 7]

        for level in levels:
            result = compute_wavelet_energy_spectrum(sample_signal_1d, level=level)
            assert result.shape[0] == 2**level

    def test_different_wavelets(self, sample_signal_1d):
        """Test energy spectrum with different wavelets."""
        wavelets = ["db4", "db8", "haar"]

        for wavelet in wavelets:
            result = compute_wavelet_energy_spectrum(
                sample_signal_1d, wavelet=wavelet, level=4
            )
            assert result.ndim == 1


class TestIntegration:
    """Integration tests combining multiple transforms."""

    def test_preemphasis_then_stft(self, sample_signal_1d):
        """Test typical pipeline: preemphasis -> STFT."""
        emphasized = apply_preemphasis(sample_signal_1d, alpha=0.97)
        spectrogram = compute_stft(emphasized, n_fft=512, hop_length=128)

        assert spectrogram.ndim == 2
        assert spectrogram.shape[0] > 0
        assert spectrogram.shape[1] > 0

    def test_preemphasis_then_wavelet(self, sample_signal_1d):
        """Test pipeline: preemphasis -> wavelet."""
        emphasized = apply_preemphasis(sample_signal_1d, alpha=0.97)
        coeffs = compute_wavelet(emphasized, level=6)

        assert coeffs.ndim == 2
        assert coeffs.shape[0] == 64

    def test_multichannel_workflow(self, sample_signal_2d):
        """Test processing each channel of multi-channel signal."""
        emphasized = apply_preemphasis(sample_signal_2d, alpha=0.97)

        # Process each channel
        spectrograms = []
        for channel in emphasized:
            spec = compute_stft(channel, n_fft=256, hop_length=64)
            spectrograms.append(spec)

        assert len(spectrograms) == 3
        for spec in spectrograms:
            assert spec.ndim == 2
