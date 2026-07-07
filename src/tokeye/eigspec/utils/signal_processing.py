"""
Signal processing utilities for eigspec package.

This module provides signal processing functions for spectral analysis and filtering:
- FFT spectral analysis and aggregation
- Digital filtering and windowing
- Coherence estimation and filtering
- Signal conditioning and preprocessing

Based on the MATLAB eigspec toolbox signal processing functions:
- fftspec.m - Multi-channel FFT spectral analysis with aggregation
- fftspec1.m - Single-channel FFT spectral analysis
- fftspecwin.m - Windowed FFT spectral analysis
- zmfftspec.m - Zero-mean FFT spectral analysis
- kdftspec.m - K-fold DFT spectral analysis
- yfilt.m - Digital filtering with boundary handling
- yintegrate.m - Numerical integration of time series
- ydecimate.m - Decimation with anti-aliasing
- yresample.m - Resampling with interpolation
- yaddgauss.m - Add Gaussian noise to signals
- oddevenupdate_emc.m - Even/odd mode coherence estimation
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union, Literal, List, Dict

import numpy as np
from numpy.typing import NDArray
from scipy import signal
from scipy.fft import fft, fftfreq
import numpy.typing as npt


@dataclass
class FFTSpectralResult:
    """
    Container for FFT spectral analysis results.
    
    Attributes:
        P: Power spectral density matrix, shape (n_frequencies, n_blocks)
        F: Frequency vector (normalized or Hz)
        T: Time vector for block centers
        nfft: FFT length used
        block_size: Block size used
        n_blocks: Number of blocks processed
        fs: Sampling frequency (if provided)
    """
    P: NDArray[np.floating]
    F: NDArray[np.floating] 
    T: NDArray[np.floating]
    nfft: int
    block_size: int
    n_blocks: int
    fs: Optional[float] = None


@dataclass
class CoherenceResult:
    """
    Container for coherence analysis results.
    
    Attributes:
        coherence: Coherence estimates over time
        time: Time vector
        frequency: Frequency vector
        phase: Phase difference (if computed)
    """
    coherence: NDArray[np.floating]
    time: NDArray[np.floating]
    frequency: NDArray[np.floating]
    phase: Optional[NDArray[np.floating]] = None


@dataclass
class AssessmentResult:
    """Result from array assessment analysis.
    
    Attributes:
        predictability: Deviation accounted for (DAF) by AR model per channel
        participation: Normalized participation/relevance of each channel
        rms_values: RMS values for each channel
        median_scores: Median values across all time blocks [DAF, participation, RMS]
        time_vector: Time vector for block centers
        channel_names: Channel names/identifiers
        block_parameters: Block size and stride used
        ar_lag: AR lag parameter used (for AR-based assessment)
    """
    predictability: npt.NDArray[np.floating]  # (n_channels, n_blocks)
    participation: npt.NDArray[np.floating]   # (n_channels, n_blocks)
    rms_values: npt.NDArray[np.floating]      # (n_channels, n_blocks)
    median_scores: npt.NDArray[np.floating]   # (n_channels, 3) - [DAF, participation, RMS]
    time_vector: npt.NDArray[np.floating]     # (n_blocks,)
    channel_names: List[str]
    block_parameters: Tuple[int, int]
    ar_lag: Optional[int] = None


@dataclass
class CorrelationAssessmentResult:
    """Result from correlation-based array assessment.
    
    Attributes:
        correlation_matrix: Cross-correlation matrix between channels
        channel_names: Channel names/identifiers
        block_parameters: Block size and stride used
        time_vector: Time vector for analysis
    """
    correlation_matrix: npt.NDArray[np.floating]  # (n_channels, n_channels)
    channel_names: List[str]
    block_parameters: Tuple[int, int]
    time_vector: npt.NDArray[np.floating]


def compute_zero_mean_spectrum(
    time_array: NDArray[np.number],
    signal_array: NDArray[np.number], 
    block_size: int,
    nfft: Union[int, Tuple[int, int]],
    reduced_dim: Optional[int] = None,
    figure_num: Optional[int] = None,
    freq_span: Optional[Tuple[float, float]] = None
) -> FFTSpectralResult:
    """
    Basic FFT-derived power spectral density plot of multivariate time-series.
    
    This is a port of the MATLAB zmfftspec function that aggregates channel FFTs
    and provides power spectral density estimates using block processing.
    
    Args:
        time_array: Time vector, shape (n_samples,)
        signal_array: Multi-channel data, shape (n_samples, n_channels)
        block_size: Block size for processing
        nfft: FFT length, or tuple (fft_length, smoothing_span)
        reduced_dim: Optional dimension reduction (< n_channels)
        figure_num: Figure number for plotting (ignored in Python version)
        freq_span: Optional frequency range for analysis
        
    Returns:
        FFTSpectralResult containing power spectral density and metadata
        
    Example:
        >>> t = np.linspace(0, 10, 1000)
        >>> y = np.sin(2*np.pi*t)[:, np.newaxis]
        >>> result = compute_zero_mean_spectrum(t, y, block_size=100, nfft=128)
        >>> print(f"PSD shape: {result.P.shape}")
    """
    if signal_array.ndim != 2:
        raise ValueError(f"signal_array must be 2D, got shape {signal_array.shape}")
    if len(time_array) != signal_array.shape[0]:
        raise ValueError("time_array length must match signal_array rows")
    
    n_samples, n_channels = signal_array.shape
    
    # Handle nfft specification
    if isinstance(nfft, (list, tuple)) and len(nfft) == 2:
        smooth_span = nfft[1]
        nfft = nfft[0]
    else:
        smooth_span = 1
        
    if nfft < block_size:
        raise ValueError("NFFT must be >= block_size")
    if nfft % 2 != 0:
        raise ValueError("NFFT should be an even number")
        
    nfft_half = nfft // 2
    
    # Calculate block parameters
    n_blocks = (n_samples - block_size) // block_size + 1
    dt = time_array[1] - time_array[0] if len(time_array) > 1 else 1.0
    fs = 1.0 / dt
    
    # Initialize output arrays
    P = np.zeros((nfft_half, n_blocks))
    T = np.zeros(n_blocks)
    
    # Frequency vector (normalized angular frequency)
    F = np.arange(nfft_half) * (2 * np.pi) / nfft
    
    # Process each block
    for j in range(n_blocks):
        start_idx = j * block_size
        end_idx = start_idx + block_size
        T[j] = time_array[start_idx + block_size // 2]  # Block center time
        
        # Extract block and compute aggregated FFT
        Y_block = signal_array[start_idx:end_idx, :]
        P[:, j] = aggregate_fft(Y_block, nfft, reduced_dim)
    
    # Apply smoothing if requested
    if smooth_span > 1:
        # Simple moving average smoothing
        kernel = np.ones(smooth_span) / smooth_span
        for i in range(nfft_half):
            P[i, :] = np.convolve(P[i, :], kernel, mode='same')
    
    return FFTSpectralResult(
        P=P, F=F, T=T, nfft=nfft, 
        block_size=block_size, n_blocks=n_blocks, fs=fs
    )


def aggregate_fft(
    signal_block: NDArray[np.number],
    nfft: int, 
    reduced_dim: Optional[int] = None
) -> NDArray[np.floating]:
    """
    Aggregate power spectrum from all channels in a signal block.
    
    Args:
        signal_block: Block of multichannel data, shape (n_samples, n_channels)
        nfft: FFT length 
        reduced_dim: Optional dimension for random projection
        
    Returns:
        Aggregated power spectrum, shape (nfft//2,)
    """
    n_samples, n_channels = signal_block.shape
    nfft_half = nfft // 2
    
    if reduced_dim is not None and reduced_dim < n_channels:
        # Apply random projection for dimension reduction
        projection_matrix = np.random.randn(reduced_dim, n_channels)
        projection_matrix /= np.linalg.norm(projection_matrix, axis=1, keepdims=True)
        signal_block = (projection_matrix @ signal_block.T).T
        n_channels = reduced_dim
    
    # Compute FFT for each channel and aggregate
    P = np.zeros(nfft_half)
    for ch in range(n_channels):
        Y_fft = fft(signal_block[:, ch], nfft)
        Y_half = Y_fft[:nfft_half]
        P += np.abs(Y_half) ** 2
        
    return P / n_channels


def filter_signal(
    signal_array: NDArray[np.number],
    filter_type: Literal['LP', 'HP', 'BP', 'BS'],
    frequency: Union[float, Tuple[float, float]],
    filter_method: Literal['filtfilt', 'filter'] = 'filtfilt',
    filter_order: int = 2
) -> NDArray[np.floating]:
    """
    Filter columns of signal array independently as time-series.
    
    Supports LP (lowpass), HP (highpass), BP (bandpass), and BS (bandstop) filters
    using Butterworth design by default.
    
    Args:
        signal_array: Input data, shape (n_samples, n_channels)
        filter_type: Type of filter ('LP', 'HP', 'BP', 'BS')
        frequency: Cutoff frequency (scalar) or band (tuple) in normalized units [0,1]
        filter_method: 'filtfilt' for zero-phase or 'filter' for causal
        filter_order: Filter order (default: 2)
        
    Returns:
        Filtered signal array, same shape as input
        
    Example:
        >>> y = np.random.randn(1000, 3)
        >>> y_filt = filter_signal(y, 'LP', 0.1)  # 10% Nyquist lowpass
    """
    if signal_array.ndim != 2:
        raise ValueError(f"signal_array must be 2D, got shape {signal_array.shape}")
        
    filter_type = filter_type.upper()
    if filter_type not in ['LP', 'HP', 'BP', 'BS']:
        raise ValueError(f"Invalid filter_type: {filter_type}")
    
    # Validate frequency specification
    if filter_type in ['BP', 'BS']:
        if not isinstance(frequency, (list, tuple)) or len(frequency) != 2:
            raise ValueError(f"{filter_type} filter requires frequency band [f1, f2]")
        if any(f <= 0 or f >= 1 for f in frequency):
            raise ValueError("Frequencies must be in range (0, 1)")
    else:
        if isinstance(frequency, (list, tuple)):
            frequency = frequency[0]
        if frequency <= 0 or frequency >= 1:
            raise ValueError("Frequency must be in range (0, 1)")
    
    # Design Butterworth filter
    if filter_type == 'LP':
        sos = signal.butter(filter_order, frequency, btype='low', output='sos')
    elif filter_type == 'HP':
        sos = signal.butter(filter_order, frequency, btype='high', output='sos')
    elif filter_type == 'BP':
        sos = signal.butter(filter_order, frequency, btype='band', output='sos')
    elif filter_type == 'BS':
        sos = signal.butter(filter_order, frequency, btype='bandstop', output='sos')
    
    # Apply filter to each channel
    n_samples, n_channels = signal_array.shape
    filtered_signal = np.zeros_like(signal_array)
    
    for ch in range(n_channels):
        if filter_method == 'filtfilt':
            filtered_signal[:, ch] = signal.sosfiltfilt(sos, signal_array[:, ch])
        else:
            filtered_signal[:, ch] = signal.sosfilt(sos, signal_array[:, ch])
            
    return filtered_signal


def create_window(
    window_type: Literal['hann', 'hamming', 'rectangular', 'blackman'],
    window_length: int
) -> NDArray[np.floating]:
    """
    Create window function for spectral analysis.
    
    Args:
        window_type: Type of window function
        window_length: Length of window
        
    Returns:
        Window function values, shape (window_length,)
    """
    if window_type == 'hann':
        return signal.windows.hann(window_length)
    elif window_type == 'hamming':
        return signal.windows.hamming(window_length)
    elif window_type == 'rectangular':
        return np.ones(window_length)
    elif window_type == 'blackman':
        return signal.windows.blackman(window_length)
    else:
        raise ValueError(f"Unknown window type: {window_type}")


def fftspec(
    time_array: NDArray[np.number],
    signal_array: NDArray[np.number],
    block_size: int,
    nfft: int,
    reduced_dim: Optional[int] = None,
    window_type: str = 'hann'
) -> FFTSpectralResult:
    """
    FFT spectral analysis with windowing and optional dimension reduction.
    
    Args:
        time_array: Time vector
        signal_array: Multi-channel signal data  
        block_size: Block size for analysis
        nfft: FFT length
        reduced_dim: Optional dimension reduction
        window_type: Window function type
        
    Returns:
        FFTSpectralResult with spectral analysis results
    """
    if signal_array.ndim != 2:
        raise ValueError(f"signal_array must be 2D, got shape {signal_array.shape}")
    
    n_samples, n_channels = signal_array.shape
    n_blocks = (n_samples - block_size) // block_size + 1
    nfft_half = nfft // 2
    
    # Create window function
    window = create_window(window_type, block_size)
    window_power = np.sum(window ** 2)
    
    # Initialize output
    P = np.zeros((nfft_half, n_blocks))
    T = np.zeros(n_blocks)
    
    # Frequency vector
    dt = time_array[1] - time_array[0] if len(time_array) > 1 else 1.0
    F = fftfreq(nfft, dt)[:nfft_half]
    
    # Process each block
    for j in range(n_blocks):
        start_idx = j * block_size
        end_idx = start_idx + block_size
        T[j] = time_array[start_idx + block_size // 2]
        
        # Extract and window the block
        block = signal_array[start_idx:end_idx, :]
        windowed_block = block * window[:, np.newaxis]
        
        # Compute power spectrum
        P[:, j] = aggregate_fft(windowed_block, nfft, reduced_dim)
        
    # Normalize by window power
    P /= window_power
    
    return FFTSpectralResult(
        P=P, F=F, T=T, nfft=nfft,
        block_size=block_size, n_blocks=n_blocks, fs=1.0/dt
    )


def coherence_filter(
    signal_x: NDArray[np.number],
    signal_y: NDArray[np.number], 
    block_size: int,
    nfft: int,
    stride: int,
    forgetting_factor: float
) -> CoherenceResult:
    """
    Coherence filter with forgetting factor for two signals.
    
    Args:
        signal_x: First signal
        signal_y: Second signal  
        block_size: Block size for analysis
        nfft: FFT length
        stride: Stride between blocks
        forgetting_factor: Forgetting factor (0 < beta < 1)
        
    Returns:
        CoherenceResult with coherence estimates over time
    """
    if len(signal_x) != len(signal_y):
        raise ValueError("Signals must have same length")
    if nfft < block_size:
        raise ValueError("nfft must be >= block_size")
    if nfft % 2 != 0:
        raise ValueError("nfft must be even")
        
    n_samples = len(signal_x)
    nfft_half = nfft // 2
    n_blocks = (n_samples - block_size) // stride + 1
    
    # Initialize
    coherence = np.zeros((nfft_half, n_blocks))
    time_vec = np.zeros(n_blocks)
    frequency = np.arange(nfft_half) * np.pi / nfft_half
    
    # Create Hann window
    window = signal.windows.hann(block_size)
    
    # Initialize filtered cross-spectral quantities
    Pxx = np.zeros(nfft_half, dtype=complex)
    Pyy = np.zeros(nfft_half, dtype=complex) 
    Pxy = np.zeros(nfft_half, dtype=complex)
    
    for j in range(n_blocks):
        start_idx = j * stride
        end_idx = start_idx + block_size
        time_vec[j] = start_idx + block_size // 2
        
        # Extract and window signals
        x_block = signal_x[start_idx:end_idx] * window
        y_block = signal_y[start_idx:end_idx] * window
        
        # Compute FFTs
        X = fft(x_block, nfft)[:nfft_half]
        Y = fft(y_block, nfft)[:nfft_half]
        
        # Update filtered quantities with forgetting factor
        Pxx = forgetting_factor * Pxx + (1 - forgetting_factor) * (X * X.conj())
        Pyy = forgetting_factor * Pyy + (1 - forgetting_factor) * (Y * Y.conj())
        Pxy = forgetting_factor * Pxy + (1 - forgetting_factor) * (X * Y.conj())
        
        # Compute coherence with numerical stability
        denominator = np.abs(Pxx) * np.abs(Pyy)
        # Avoid division by zero and ensure coherence <= 1
        coherence[:, j] = np.minimum(1.0, np.abs(Pxy) ** 2 / np.maximum(denominator, 1e-15))
        
    return CoherenceResult(
        coherence=coherence, time=time_vec, 
        frequency=frequency, phase=np.angle(Pxy)
    ) 


def ar_assessment(
    time_vector: Optional[npt.NDArray[np.floating]],
    signal_array: npt.NDArray[np.floating],
    ar_lag: int,
    block_parameters: Tuple[int, int],
    median_filter_length: int = 0,
    channel_names: Optional[List[str]] = None
) -> AssessmentResult:
    """AR-based array assessment for detecting problematic channels.
    
    Python equivalent of arassess.m
    
    Analyzes each channel's predictability using autoregressive modeling
    and participation in the overall array response. Useful for identifying
    faulty sensors or channels with poor signal quality.
    
    Args:
        time_vector: Time vector or None (uses indices if None)
        signal_array: Input data matrix (n_samples, n_channels)
        ar_lag: AR model lag (past samples to use for prediction)
        block_parameters: (block_size, block_stride) for analysis
        median_filter_length: Length of median filter preprocessing (0 = none)
        channel_names: Optional channel names (generated if None)
        
    Returns:
        AssessmentResult with channel health metrics
        
    Raises:
        ValueError: If inputs are malformed or inconsistent
    """
    if signal_array.ndim != 2:
        raise ValueError(f"signal_array must be 2D, got shape {signal_array.shape}")
    
    n_samples, n_channels = signal_array.shape
    
    if time_vector is not None:
        time_vector = np.asarray(time_vector)
        if len(time_vector) != n_samples:
            raise ValueError("Length of time_vector must match signal_array first dimension")
    
    if ar_lag < 1:
        raise ValueError("ar_lag must be positive")
    
    block_size, block_stride = block_parameters
    if block_size <= ar_lag:
        raise ValueError("block_size must be larger than ar_lag")
    
    # Generate channel names if not provided
    if channel_names is None:
        channel_names = [f"#{i+1}" for i in range(n_channels)]
    elif len(channel_names) != n_channels:
        raise ValueError("Length of channel_names must match number of channels")
    
    # Add channel indices to names for reference
    channel_names = [f"{name} (#{i+1})" for i, name in enumerate(channel_names)]
    
    # Normalize channels to unit RMS
    Y = signal_array.copy()
    channel_rms = np.sqrt(np.mean(Y**2, axis=0))
    
    # Avoid division by zero
    nonzero_rms = channel_rms > 0
    Y[:, nonzero_rms] = Y[:, nonzero_rms] / channel_rms[nonzero_rms]
    
    if not np.all(nonzero_rms):
        print(f"Warning: {np.sum(~nonzero_rms)} channels have zero RMS")
    
    # Check for rank deficiency
    try:
        cov_matrix = np.cov(Y.T)
        rank = np.linalg.matrix_rank(cov_matrix)
        if rank < n_channels:
            print(f"Warning: signal data is not full rank (rank={rank}, channels={n_channels})")
    except np.linalg.LinAlgError:
        print("Warning: could not compute covariance matrix rank")
    
    # Apply median filtering if requested
    if median_filter_length >= 3:
        from scipy.signal import medfilt
        for ch in range(n_channels):
            Y[:, ch] = medfilt(Y[:, ch], kernel_size=median_filter_length)
    
    # Block analysis setup
    block_starts = np.arange(0, n_samples - block_size + 1, block_stride)
    n_blocks = len(block_starts)
    
    if n_blocks == 0:
        raise ValueError("No complete blocks can be formed with given parameters")
    
    # Initialize results
    predictability = np.zeros((n_channels, n_blocks))  # Deviation Accounted For (DAF)
    participation = np.zeros((n_channels, n_blocks))   # Channel participation/relevance
    rms_block_values = np.zeros((n_channels, n_blocks))  # Block RMS values
    
    if time_vector is not None:
        time_centers = np.zeros(n_blocks)
    else:
        time_centers = np.arange(n_blocks)
    
    for block_idx, block_start in enumerate(block_starts):
        block_end = block_start + block_size
        
        if time_vector is not None:
            time_centers[block_idx] = (time_vector[block_start] + time_vector[block_end-1]) / 2
        
        # Extract block data
        block_data = Y[block_start:block_end, :]
        
        # Compute AR model using multivariate approach
        try:
            ar_matrices, cov_innovation, cov_data = _multivariate_ar_model(block_data, ar_lag)
            
            # Compute predictability (Deviation Accounted For)
            innovation_var = np.diag(cov_innovation)
            data_var = np.diag(cov_data)
            
            # Avoid division by zero
            valid_vars = data_var > 0
            daf = np.zeros(n_channels)
            daf[valid_vars] = np.maximum(0, 1 - np.sqrt(innovation_var[valid_vars] / data_var[valid_vars]))
            predictability[:, block_idx] = daf
            
            # Compute participation (contribution to AR model)
            participation_scores = np.zeros(n_channels)
            for ch in range(n_channels):
                # Sum of L2 norms across all AR matrices for this channel
                channel_contrib = 0.0
                for lag in range(ar_lag):
                    # Use default norm (L2) for vectors, not Frobenius norm
                    channel_contrib += np.linalg.norm(ar_matrices[lag][:, ch])**2
                participation_scores[ch] = np.sqrt(channel_contrib)
            
            # Normalize participation
            total_participation = np.sum(participation_scores)
            if total_participation > 0:
                participation[:, block_idx] = n_channels * participation_scores / total_participation
            else:
                participation[:, block_idx] = np.ones(n_channels)  # Equal participation if all zero
                
        except (np.linalg.LinAlgError, ValueError) as e:
            print(f"Warning: AR model failed for block {block_idx}: {e}")
            # Set default values for failed blocks
            predictability[:, block_idx] = 0.0
            participation[:, block_idx] = 1.0
        
        # Store RMS values (scaled back to original units)
        block_rms = np.sqrt(np.mean(block_data**2, axis=0))
        rms_block_values[:, block_idx] = block_rms * channel_rms
    
    # Compute median scores across all blocks
    median_scores = np.column_stack([
        np.median(predictability, axis=1),
        np.median(participation, axis=1),
        np.median(rms_block_values, axis=1)
    ])
    
    return AssessmentResult(
        predictability=predictability,
        participation=participation,
        rms_values=rms_block_values,
        median_scores=median_scores,
        time_vector=time_centers,
        channel_names=channel_names,
        block_parameters=block_parameters,
        ar_lag=ar_lag
    )


def correlation_assessment(
    time_vector: Optional[npt.NDArray[np.floating]],
    signal_array: npt.NDArray[np.floating],
    block_parameters: Tuple[int, int],
    channel_names: Optional[List[str]] = None
) -> CorrelationAssessmentResult:
    """Simple correlation-based array assessment.
    
    Python equivalent of corrassess.m
    
    Computes cross-correlation matrix between all channels to assess
    array coherence and identify outlier channels.
    
    Args:
        time_vector: Time vector or None (uses indices if None)
        signal_array: Input data matrix (n_samples, n_channels)
        block_parameters: (block_size, block_stride) for analysis
        channel_names: Optional channel names (generated if None)
        
    Returns:
        CorrelationAssessmentResult with correlation matrix
        
    Raises:
        ValueError: If inputs are malformed
    """
    if signal_array.ndim != 2:
        raise ValueError(f"signal_array must be 2D, got shape {signal_array.shape}")
    
    n_samples, n_channels = signal_array.shape
    
    if time_vector is not None:
        time_vector = np.asarray(time_vector)
        if len(time_vector) != n_samples:
            raise ValueError("Length of time_vector must match signal_array first dimension")
    else:
        time_vector = np.arange(n_samples)
    
    # Generate channel names if not provided
    if channel_names is None:
        channel_names = [f"#{i+1}" for i in range(n_channels)]
    elif len(channel_names) != n_channels:
        raise ValueError("Length of channel_names must match number of channels")
    
    block_size, block_stride = block_parameters
    
    # Use middle section of data for correlation analysis
    if block_size < n_samples:
        start_idx = (n_samples - block_size) // 2
        end_idx = start_idx + block_size
        analysis_data = signal_array[start_idx:end_idx, :]
        analysis_time = time_vector[start_idx:end_idx]
    else:
        analysis_data = signal_array
        analysis_time = time_vector
    
    print(f"Computing correlation matrix for {n_channels} channels using {len(analysis_data)} samples")
    
    # Compute correlation matrix with warning suppression for constant signals
    with np.errstate(divide='ignore', invalid='ignore'):
        correlation_matrix = np.corrcoef(analysis_data.T)
    
    # Handle NaN values (can occur with constant signals)
    if np.any(np.isnan(correlation_matrix)):
        print("Warning: NaN values detected in correlation matrix (replacing with zeros)")
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
    
    return CorrelationAssessmentResult(
        correlation_matrix=correlation_matrix,
        channel_names=channel_names,
        block_parameters=block_parameters,
        time_vector=analysis_time
    )


def _multivariate_ar_model(
    data: npt.NDArray[np.floating], 
    lag: int
) -> Tuple[List[npt.NDArray[np.floating]], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Fit multivariate AR model to data.
    
    Python equivalent of the subvar function in arassess.m
    
    Args:
        data: Input data matrix (n_samples, n_channels) 
        lag: AR model lag order
        
    Returns:
        Tuple of (ar_matrices, innovation_covariance, data_covariance)
        - ar_matrices: List of AR coefficient matrices [A1, A2, ..., Ap]
        - innovation_covariance: Covariance of prediction errors
        - data_covariance: Covariance of input data
        
    Raises:
        ValueError: If data is insufficient or ill-conditioned
    """
    n_samples, n_channels = data.shape
    
    if n_samples <= lag:
        raise ValueError(f"Need more samples than lag: {n_samples} <= {lag}")
    
    n_effective = n_samples - lag
    n_features = n_channels * lag
    
    # Build regressor matrix (past observations) and response matrix (current observations)
    Z = data.T  # Transpose for easier indexing
    regressor_matrix = np.zeros((n_features, n_effective))
    response_matrix = np.zeros((n_channels, n_effective))
    
    for k in range(lag, n_samples):
        # Stack past lag observations into regressor vector
        past_obs = []
        for lag_idx in range(lag):
            past_obs.append(Z[:, k - lag_idx - 1])
        regressor_vector = np.concatenate(past_obs)
        
        regressor_matrix[:, k - lag] = regressor_vector
        response_matrix[:, k - lag] = data[k, :]
    
    # Solve normal equations: Y = H * Zp, where H are AR coefficients
    try:
        regressor_cov = regressor_matrix @ regressor_matrix.T / n_effective
        cross_cov = response_matrix @ regressor_matrix.T / n_effective
        
        # Solve for AR coefficient matrix H
        ar_coeff_matrix = cross_cov @ np.linalg.pinv(regressor_cov)
        
    except np.linalg.LinAlgError:
        raise ValueError("Failed to solve AR normal equations (singular covariance matrix)")
    
    # Split AR coefficient matrix into per-lag matrices
    ar_matrices = []
    for lag_idx in range(lag):
        start_idx = lag_idx * n_channels
        end_idx = (lag_idx + 1) * n_channels
        ar_matrices.append(ar_coeff_matrix[:, start_idx:end_idx])
    
    # Compute covariances
    data_covariance = response_matrix @ response_matrix.T / n_effective
    
    # Prediction errors
    prediction_errors = response_matrix - ar_coeff_matrix @ regressor_matrix
    innovation_covariance = prediction_errors @ prediction_errors.T / n_effective
    
    return ar_matrices, innovation_covariance, data_covariance 


try:
    from numba import jit, prange
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    
    # Fallback decorator that does nothing
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def prange(x):
        return range(x)

try:
    from joblib import Parallel, delayed
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False
    
    # Fallback implementations
    def delayed(func):
        return func
    
    class Parallel:
        def __init__(self, *args, **kwargs):
            pass
        
        def __call__(self, iterable):
            return list(iterable)


@jit(nopython=True, cache=True) if _NUMBA_AVAILABLE else lambda x: x
def _fast_fft_magnitude_squared(real_part: npt.NDArray[np.floating], 
                               imag_part: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Optimized computation of FFT magnitude squared.
    
    Args:
        real_part: Real part of FFT
        imag_part: Imaginary part of FFT
        
    Returns:
        Magnitude squared values
    """
    result = np.zeros_like(real_part)
    for i in prange(len(real_part)):
        result[i] = real_part[i]**2 + imag_part[i]**2
    return result


@jit(nopython=True, cache=True) if _NUMBA_AVAILABLE else lambda x: x
def _fast_mac_computation(shape1_real: npt.NDArray[np.floating],
                         shape1_imag: npt.NDArray[np.floating],
                         shape2_real: npt.NDArray[np.floating], 
                         shape2_imag: npt.NDArray[np.floating]) -> float:
    """Optimized MAC computation for complex vectors.
    
    Args:
        shape1_real: Real part of first shape vector
        shape1_imag: Imaginary part of first shape vector
        shape2_real: Real part of second shape vector
        shape2_imag: Imaginary part of second shape vector
        
    Returns:
        MAC value
    """
    # Compute complex dot products
    dot_12_real = 0.0
    dot_12_imag = 0.0
    norm1_sq = 0.0
    norm2_sq = 0.0
    
    for i in range(len(shape1_real)):
        # Conjugate of shape1 dot shape2
        dot_12_real += shape1_real[i] * shape2_real[i] + shape1_imag[i] * shape2_imag[i]
        dot_12_imag += shape1_real[i] * shape2_imag[i] - shape1_imag[i] * shape2_real[i]
        
        # Norms squared
        norm1_sq += shape1_real[i]**2 + shape1_imag[i]**2
        norm2_sq += shape2_real[i]**2 + shape2_imag[i]**2
    
    # MAC = |dot_12|^2 / (norm1_sq * norm2_sq)
    dot_12_mag_sq = dot_12_real**2 + dot_12_imag**2
    
    if norm1_sq * norm2_sq == 0.0:
        return 0.0
    
    return dot_12_mag_sq / (norm1_sq * norm2_sq)


@jit(nopython=True, cache=True) if _NUMBA_AVAILABLE else lambda x: x  
def _fast_correlation_matrix(data: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Optimized correlation matrix computation.
    
    Args:
        data: Data matrix (n_samples, n_channels)
        
    Returns:
        Correlation matrix (n_channels, n_channels)
    """
    n_samples, n_channels = data.shape
    
    # Compute means
    means = np.zeros(n_channels)
    for j in range(n_channels):
        for i in range(n_samples):
            means[j] += data[i, j]
        means[j] /= n_samples
    
    # Compute standard deviations
    stds = np.zeros(n_channels)
    for j in range(n_channels):
        for i in range(n_samples):
            diff = data[i, j] - means[j]
            stds[j] += diff * diff
        stds[j] = np.sqrt(stds[j] / (n_samples - 1))
    
    # Compute correlation matrix
    corr_matrix = np.zeros((n_channels, n_channels))
    
    for i in range(n_channels):
        for j in range(i, n_channels):
            if stds[i] == 0.0 or stds[j] == 0.0:
                corr_matrix[i, j] = 0.0
            else:
                covariance = 0.0
                for k in range(n_samples):
                    covariance += (data[k, i] - means[i]) * (data[k, j] - means[j])
                covariance /= (n_samples - 1)
                
                correlation = covariance / (stds[i] * stds[j])
                corr_matrix[i, j] = correlation
                corr_matrix[j, i] = correlation  # Symmetric matrix
    
    return corr_matrix


def parallel_block_processing(
    data: npt.NDArray[np.floating],
    block_function: callable,
    block_parameters: Tuple[int, int],
    n_jobs: int = -1,
    **kwargs
) -> List:
    """Process data blocks in parallel.
    
    Args:
        data: Input data matrix (n_samples, n_channels)
        block_function: Function to apply to each block
        block_parameters: (block_size, block_stride)
        n_jobs: Number of parallel jobs (-1 for all cores)
        **kwargs: Additional arguments for block_function
        
    Returns:
        List of results from each block
    """
    if not _JOBLIB_AVAILABLE:
        print("Warning: joblib not available, falling back to sequential processing")
    
    n_samples = data.shape[0]
    block_size, block_stride = block_parameters
    
    block_starts = np.arange(0, n_samples - block_size + 1, block_stride)
    
    def process_block(start_idx):
        end_idx = start_idx + block_size
        block_data = data[start_idx:end_idx]
        return block_function(block_data, **kwargs)
    
    if _JOBLIB_AVAILABLE and n_jobs != 1:
        # Parallel processing
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_block)(start_idx) for start_idx in block_starts
        )
    else:
        # Sequential processing
        results = [process_block(start_idx) for start_idx in block_starts]
    
    return results


def optimized_spectral_analysis(
    signal_data: npt.NDArray[np.floating],
    fft_size: int = 2048,
    overlap: float = 0.5,
    window: str = "hann",
    n_jobs: int = -1
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Optimized parallel spectral analysis.
    
    Args:
        signal_data: Input signals (n_samples, n_channels)
        fft_size: FFT size for analysis
        overlap: Overlap fraction between segments
        window: Window function name
        n_jobs: Number of parallel jobs
        
    Returns:
        Tuple of (frequencies, times, power_spectral_density)
    """
    n_samples, n_channels = signal_data.shape
    hop_size = int(fft_size * (1 - overlap))
    
    # Generate window
    if window == "hann":
        win = np.hanning(fft_size)
    elif window == "hamming":
        win = np.hamming(fft_size)
    elif window == "blackman":
        win = np.blackman(fft_size)
    else:
        win = np.ones(fft_size)  # Rectangular
    
    # Normalize window
    win_norm = np.sum(win**2)
    
    # Block processing function
    def compute_segment_psd(data_segment):
        if len(data_segment) < fft_size:
            # Pad with zeros
            padded = np.zeros((fft_size, n_channels))
            padded[:len(data_segment)] = data_segment
            data_segment = padded
        
        # Apply window and compute FFT for all channels
        windowed = data_segment * win[:, np.newaxis]
        fft_result = fft(windowed, axis=0)
        
        # Compute power spectral density
        if _NUMBA_AVAILABLE:
            psd = np.zeros((fft_size, n_channels))
            for ch in range(n_channels):
                psd[:, ch] = _fast_fft_magnitude_squared(
                    fft_result[:, ch].real, fft_result[:, ch].imag
                )
        else:
            psd = np.abs(fft_result)**2
        
        # Normalize
        psd = psd / (win_norm * n_samples)
        
        return psd
    
    # Parallel processing of segments
    segment_starts = np.arange(0, n_samples - fft_size + 1, hop_size)
    
    if _JOBLIB_AVAILABLE and n_jobs != 1:
        psd_segments = Parallel(n_jobs=n_jobs)(
            delayed(compute_segment_psd)(signal_data[start:start+fft_size])
            for start in segment_starts
        )
    else:
        psd_segments = [
            compute_segment_psd(signal_data[start:start+fft_size])
            for start in segment_starts
        ]
    
    # Combine results
    psd_array = np.stack(psd_segments, axis=2)  # (freq, channels, time)
    
    # Generate frequency and time axes
    frequencies = fftfreq(fft_size, d=1.0)[:fft_size//2]  # Positive frequencies only
    times = segment_starts / n_samples  # Normalized time
    
    # Return only positive frequencies
    psd_positive = psd_array[:fft_size//2, :, :]
    
    return frequencies, times, psd_positive


def optimized_ar_assessment(
    signal_data: npt.NDArray[np.floating],
    ar_lag: int,
    block_parameters: Tuple[int, int],
    n_jobs: int = -1
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Optimized parallel AR-based assessment.
    
    Args:
        signal_data: Input signals (n_samples, n_channels)
        ar_lag: AR model lag order
        block_parameters: (block_size, block_stride)
        n_jobs: Number of parallel jobs
        
    Returns:
        Tuple of (predictability_matrix, participation_matrix)
    """
    def ar_block_analysis(block_data):
        """Analyze single block for AR metrics."""
        try:
            ar_matrices, cov_innovation, cov_data = _multivariate_ar_model(block_data, ar_lag)
            
            # Compute predictability
            innovation_var = np.diag(cov_innovation)
            data_var = np.diag(cov_data)
            
            valid_vars = data_var > 0
            daf = np.zeros(len(data_var))
            daf[valid_vars] = np.maximum(0, 1 - np.sqrt(innovation_var[valid_vars] / data_var[valid_vars]))
            
            # Compute participation
            n_channels = block_data.shape[1]
            participation_scores = np.zeros(n_channels)
            
            for ch in range(n_channels):
                channel_contrib = 0.0
                for lag_idx in range(ar_lag):
                    channel_contrib += np.linalg.norm(ar_matrices[lag_idx][:, ch])**2
                participation_scores[ch] = np.sqrt(channel_contrib)
            
            # Normalize participation
            total_participation = np.sum(participation_scores)
            if total_participation > 0:
                participation_scores = n_channels * participation_scores / total_participation
            else:
                participation_scores = np.ones(n_channels)
            
            return daf, participation_scores
            
        except (np.linalg.LinAlgError, ValueError):
            # Return zeros for failed blocks
            n_channels = block_data.shape[1]
            return np.zeros(n_channels), np.ones(n_channels)
    
    # Use parallel block processing
    results = parallel_block_processing(
        signal_data,
        ar_block_analysis,
        block_parameters,
        n_jobs=n_jobs
    )
    
    # Combine results
    if results:
        predictability_list, participation_list = zip(*results)
        predictability_matrix = np.column_stack(predictability_list)
        participation_matrix = np.column_stack(participation_list)
    else:
        n_channels = signal_data.shape[1]
        predictability_matrix = np.array([]).reshape(n_channels, 0)
        participation_matrix = np.array([]).reshape(n_channels, 0)
    
    return predictability_matrix, participation_matrix


def batch_mac_computation(
    reference_shapes: npt.NDArray[np.complexfloating],
    test_shapes: npt.NDArray[np.complexfloating],
    n_jobs: int = -1
) -> npt.NDArray[np.floating]:
    """Compute MAC values between reference and test shapes in parallel.
    
    Args:
        reference_shapes: Reference mode shapes (n_dofs, n_ref_modes)
        test_shapes: Test mode shapes (n_dofs, n_test_modes)
        n_jobs: Number of parallel jobs
        
    Returns:
        MAC matrix (n_ref_modes, n_test_modes)
    """
    n_ref_modes = reference_shapes.shape[1]
    n_test_modes = test_shapes.shape[1]
    
    def compute_mac_row(ref_idx):
        """Compute MAC values for one reference mode against all test modes."""
        ref_shape = reference_shapes[:, ref_idx]
        mac_row = np.zeros(n_test_modes)
        
        if _NUMBA_AVAILABLE:
            ref_real = ref_shape.real
            ref_imag = ref_shape.imag
            
            for test_idx in range(n_test_modes):
                test_shape = test_shapes[:, test_idx]
                mac_row[test_idx] = _fast_mac_computation(
                    ref_real, ref_imag, test_shape.real, test_shape.imag
                )
        else:
            for test_idx in range(n_test_modes):
                test_shape = test_shapes[:, test_idx]
                
                numerator = (ref_shape.conj().T @ test_shape) * (test_shape.conj().T @ ref_shape)
                denominator = (ref_shape.conj().T @ ref_shape) * (test_shape.conj().T @ test_shape)
                
                if np.abs(denominator) == 0:
                    mac_row[test_idx] = 0.0
                else:
                    mac_row[test_idx] = np.real(numerator / denominator)
        
        return mac_row
    
    if _JOBLIB_AVAILABLE and n_jobs != 1:
        mac_rows = Parallel(n_jobs=n_jobs)(
            delayed(compute_mac_row)(ref_idx) for ref_idx in range(n_ref_modes)
        )
    else:
        mac_rows = [compute_mac_row(ref_idx) for ref_idx in range(n_ref_modes)]
    
    return np.array(mac_rows)


def get_performance_info() -> Dict[str, bool]:
    """Get information about available performance optimizations.
    
    Returns:
        Dictionary with availability of optimization libraries
    """
    return {
        "numba_available": _NUMBA_AVAILABLE,
        "joblib_available": _JOBLIB_AVAILABLE,
        "parallel_processing": _JOBLIB_AVAILABLE,
        "jit_compilation": _NUMBA_AVAILABLE
    }


# =============================================================================
# MATLAB Assessment Functions Implementation (arassess.m and corrassess.m)
# =============================================================================

def arassess(
    time_vector: Optional[npt.NDArray[np.floating]],
    signal_data: npt.NDArray[np.floating],
    ar_order: int,
    block_params: Tuple[int, int],
    median_filter_length: Optional[int] = None,
    channel_names: Optional[List[str]] = None
) -> AssessmentResult:
    """
    Auto-detect problematic channels using AR modeling predictability and participation.
    
    Direct Python port of MATLAB arassess.m function for analyzing channel quality through:
    - Predictability (DAF): How well each channel can be predicted from AR model  
    - Participation (PRT): How much each channel contributes to AR model
    - RMS levels: Signal power levels
    
    Args:
        time_vector: Time vector, shape (N,) or None for sample indices
        signal_data: Signal data matrix, shape (N, M) where M is channels
        ar_order: AR model order (scalar, past lag horizon p)
        block_params: Block parameters (block_size, block_stride)
        median_filter_length: Optional median filter length (>=3 to apply)
        channel_names: Optional channel names list
        
    Returns:
        AssessmentResult containing channel health assessment
    """
    N, M = signal_data.shape
    block_size, block_stride = block_params
    
    # Normalize channels to unity RMS (following MATLAB arassess.m exactly)
    Y = signal_data.copy().astype(np.float64)
    
    # Compute RY matrix and RMS values exactly as in MATLAB
    RY = (Y.T @ Y) / N
    rmsy = np.sqrt(np.diag(RY))
    
    # Normalize each channel by its RMS
    for cc in range(M):
        if rmsy[cc] > 0:
            Y[:, cc] = Y[:, cc] / rmsy[cc]
    
    # Check rank of covariance matrix
    if np.linalg.matrix_rank(RY) != M:
        print("Warning: signal data is not full rank")
    
    # Apply median filtering if requested (MATLAB logic)
    if median_filter_length is not None and median_filter_length >= 3:
        from scipy.signal import medfilt
        for cc in range(M):
            Y[:, cc] = medfilt(Y[:, cc], kernel_size=median_filter_length)
    
    # Set up channel names matching MATLAB format
    if channel_names is None:
        channel_names = [f"#{cc+1}" for cc in range(M)]
    else:
        if len(channel_names) != M:
            raise ValueError("Channel names length must match number of channels")
        # Add channel numbers like MATLAB version
        channel_names = [f"{name} (#{cc+1})" for cc, name in enumerate(channel_names)]
    
    # Block analysis setup (MATLAB variable names)
    t1vec = np.arange(0, N - block_size + 1, block_stride)
    NBlock = len(t1vec)
    
    if NBlock == 0:
        raise ValueError("No complete blocks can be formed with given parameters")
    
    # Time vector for blocks  
    if time_vector is not None and len(time_vector) > 0:
        t = np.zeros(NBlock)
        for jj in range(NBlock):
            t1, t2 = t1vec[jj], t1vec[jj] + block_size - 1
            t[jj] = (time_vector[t1] + time_vector[t2]) / 2
    else:
        t = np.arange(1, NBlock + 1)  # MATLAB 1-based indexing style
    
    # Initialize results matrices (matching MATLAB variable names)
    DAF = np.zeros((M, NBlock))  # Deviation Accounted For (predictability)
    PRT = np.zeros((M, NBlock))  # Participation/relevance 
    RMS = np.zeros((M, NBlock))  # RMS levels
    
    for jj in range(NBlock):
        t1 = t1vec[jj]
        t2 = t1 + block_size
        
        # Process block - fit AR model using subvar equivalent
        try:
            block_data = Y[t1:t2, :]
            H, Ry, Re = _subvar_matlab(block_data, ar_order)
            
            # Deviation accounted for (predictability) - exact MATLAB formula
            dafjj = np.maximum(0, 1 - np.sqrt(np.diag(Re) / np.diag(Ry)))
            
            # Participation for each channel - exact MATLAB logic
            prtjj = np.zeros(M)
            for ii in range(M):
                # Extract H coefficients for channel ii across all lags: H(:,ii:M:(p*M))
                # H is shape (M, M * ar_order), channel ii appears at positions ii, ii+M, ii+2*M, etc.
                channel_cols = np.arange(ii, M * ar_order, M)
                prtjj[ii] = np.linalg.norm(H[:, channel_cols], 'fro')
            
            # Normalize participation to sum to M (exact MATLAB formula: M*prtjj/sum(prtjj))
            prt_sum = np.sum(prtjj)
            if prt_sum > 0:
                prtjj = M * prtjj / prt_sum
            
            # Store results
            DAF[:, jj] = dafjj
            PRT[:, jj] = prtjj
            RMS[:, jj] = np.sqrt(np.diag(Ry)) * rmsy
            
        except (np.linalg.LinAlgError, ValueError) as e:
            print(f"Warning: AR model failed for block {jj}: {e}")
            # Set default values for failed blocks
            DAF[:, jj] = 0.0
            PRT[:, jj] = 1.0
            RMS[:, jj] = rmsy
    
    # Compute median values (meddpr in MATLAB)
    meddpr = np.column_stack([
        np.median(DAF, axis=1),  # Median predictability
        np.median(PRT, axis=1),  # Median participation  
        np.median(RMS, axis=1)   # Median RMS
    ])
    
    return AssessmentResult(
        predictability=DAF,
        participation=PRT, 
        rms_values=RMS,
        median_scores=meddpr,
        time_vector=t,
        channel_names=channel_names,
        block_parameters=block_params,
        ar_lag=ar_order
    )


def corrassess(
    time_vector: Optional[npt.NDArray[np.floating]],
    signal_data: npt.NDArray[np.floating],
    block_params: Tuple[int, int],
    channel_names: Optional[List[str]] = None
) -> CorrelationAssessmentResult:
    """
    Simple correlation map output to assess channel correlation patterns.
    
    Direct Python port of MATLAB corrassess.m for analyzing correlations between channels
    to identify problematic or redundant channels.
    
    Args:
        time_vector: Time vector, shape (N,) or None for sample indices
        signal_data: Signal data matrix, shape (N, M) where M is number of channels
        block_params: Block parameters (block_size, block_stride)
        channel_names: Optional channel names list
        
    Returns:
        CorrelationAssessmentResult containing correlation analysis results
    """
    N, M = signal_data.shape
    block_size, block_stride = block_params
    
    # Check data rank (following MATLAB corrassess.m)
    RY = (signal_data.T @ signal_data) / N
    if np.linalg.matrix_rank(RY) != M:
        print("Warning: signal data is not full rank")
    
    # Set up channel names matching MATLAB format
    if channel_names is None:
        channel_names = [f"#{cc+1}" for cc in range(M)]
    else:
        if len(channel_names) != M:
            raise ValueError("Channel names length must match number of channels")
        # Add channel numbers like MATLAB version  
        channel_names = [f"{name} (#{cc+1})" for cc, name in enumerate(channel_names)]
    
    # Block analysis setup (MATLAB variable names)
    t1vec = np.arange(0, N - block_size + 1, block_stride)
    NBlock = len(t1vec)
    
    if NBlock == 0:
        raise ValueError("No complete blocks can be formed with given parameters")
    
    # Time vector for blocks
    if time_vector is not None and len(time_vector) > 0:
        t = np.zeros(NBlock)
        for jj in range(NBlock):
            t1, t2 = t1vec[jj], t1vec[jj] + block_size - 1
            t[jj] = (time_vector[t1] + time_vector[t2]) / 2
    else:
        t = np.arange(1, NBlock + 1)  # MATLAB 1-based indexing style
    
    # Correlation map
    CMAP = np.zeros((M, NBlock))
    
    for jj in range(NBlock):
        t1 = t1vec[jj]
        t2 = t1 + block_size
        
        # Process block
        Z = signal_data[t1:t2, :]
        C = np.abs(_corrloc_matlab(Z))
        
        # Average correlation with other channels: c=(sum(C,2)-diag(C))/(M-1)
        c = (np.sum(C, axis=1) - np.diag(C)) / (M - 1)
        CMAP[:, jj] = c
    
    # Median correlation across blocks: C=median(CMAP,2)
    median_correlation = np.median(CMAP, axis=1)
    
    return CorrelationAssessmentResult(
        median_correlation=median_correlation,
        correlation_map=CMAP,
        time_blocks=t,
        channel_names=channel_names
    )


def _subvar_matlab(y: npt.NDArray[np.floating], p: int) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Fit multivariate AR model using least squares.
    
    Direct Python translation of the subvar function from arassess.m
    
    Args:
        y: Signal data, shape (N, ny) 
        p: AR model order (number of lags)
        
    Returns:
        H: AR coefficient matrix, shape (ny, ny*p)
        Ry: Signal covariance matrix, shape (ny, ny)
        Re: Residual covariance matrix, shape (ny, ny)
    """
    N, ny = y.shape
    Nk = N - p  # Number of effective data points
    nyp = ny * p  # Total number of AR parameters per equation
    
    # Create matrices following MATLAB indexing
    Z = y.T  # Transpose for easier column access
    Zp = np.zeros((nyp, Nk))  # Past data matrix
    Y = np.zeros((ny, Nk))    # Current data matrix
    
    # Build lagged data matrix exactly as in MATLAB
    for kk in range(p, N):  # MATLAB: for kk=(p+1):N
        # Extract past p samples: (kk-1):-1:(kk-p) in MATLAB 1-indexing
        # zp=reshape(Z(:,(kk-1):-1:(kk-p)),nyp,1);
        past_cols = []
        for lag_idx in range(p):
            past_cols.append(Z[:, kk - 1 - lag_idx])
        
        # Reshape and store (MATLAB: reshape(..., nyp, 1))
        zp = np.concatenate(past_cols)
        Zp[:, kk - p] = zp
        Y[:, kk - p] = y[kk, :].T
    
    # Solve AR equations: Y = H * Zp + E
    # MATLAB: Rzp=Zp*Zp'; H=(Y*Zp')/Rzp;
    Rzp = Zp @ Zp.T
    H = (Y @ Zp.T) @ np.linalg.pinv(Rzp)  # AR coefficient matrix
    
    # Compute covariances (exact MATLAB formulas)
    Ry = (Y @ Y.T) / Nk         # Signal covariance: Ry=(Y*Y')/Nk;
    E = Y - H @ Zp              # Residuals: E=Y-H*Zp;
    Re = (E @ E.T) / Nk         # Residual covariance: Re=(E*E')/Nk;
    
    return H, Ry, Re


def _corrloc_matlab(Z: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """
    Compute local correlation matrix for a data block.
    
    Direct Python translation of corrloc function from corrassess.m
    
    Args:
        Z: Data block, shape (n, m)
        
    Returns:
        r: Correlation matrix, shape (m, m)
    """
    n, m = Z.shape
    
    # Normalize each column (MATLAB loop logic)
    Z_norm = np.zeros_like(Z)
    for jj in range(m):
        # MATLAB: Z(:,jj)=Z(:,jj)-mean(Z(:,jj)); Z(:,jj)=Z(:,jj)/norm(Z(:,jj));
        col = Z[:, jj] - np.mean(Z[:, jj])
        norm_col = np.linalg.norm(col)
        if norm_col > 0:
            Z_norm[:, jj] = col / norm_col
        else:
            Z_norm[:, jj] = col
    
    # MATLAB: r=(Z'*Z);
    r = Z_norm.T @ Z_norm
    return r


# =============================================================================
# Additional MATLAB Signal Processing Functions
# =============================================================================

def yfilt(
    signal_data: npt.NDArray[np.floating],
    Ts: Optional[float] = None,
    filter_type: Literal["LP", "HP", "BP", "BS"] = "LP",
    frequency_band: Union[float, Tuple[float, float]] = 0.1,
    filter_order: int = 2,
    use_filtfilt: bool = True
) -> npt.NDArray[np.floating]:
    """
    Filter columns of signal data independently as time-series.
    
    Python port of MATLAB yfilt.m that applies Butterworth filters to
    multichannel signals with zero-phase distortion option.
    
    Args:
        signal_data: Signal data where each column is a channel, shape (N, M)
        Ts: Sample period (None for normalized frequencies)  
        filter_type: Filter type ("LP", "HP", "BP", "BS")
        frequency_band: Cutoff frequency (scalar) or band (tuple)
        filter_order: Filter order (default 2)
        use_filtfilt: Use zero-phase filtering (default True)
        
    Returns:
        Filtered signal data, same shape as input
    """
    from scipy.signal import butter, filtfilt, lfilter
    
    N, M = signal_data.shape
    
    # Handle frequency normalization
    if Ts is not None:
        Fs = 1.0 / Ts
        if isinstance(frequency_band, tuple):
            fband = np.array(frequency_band) * (2.0 / Fs)
        else:
            fband = frequency_band * (2.0 / Fs)
    else:
        fband = frequency_band
    
    # Check Nyquist constraint
    if isinstance(fband, np.ndarray):
        if np.any(fband >= 1.0):
            raise ValueError("Frequencies specified beyond Nyquist limit")
    else:
        if fband >= 1.0:
            raise ValueError("Frequency specified beyond Nyquist limit")
    
    # Design filter
    if filter_type.upper() == "LP":
        b, a = butter(filter_order, fband, btype='low')
    elif filter_type.upper() == "HP":
        b, a = butter(filter_order, fband, btype='high')
    elif filter_type.upper() == "BP":
        b, a = butter(filter_order, fband, btype='band')
    elif filter_type.upper() == "BS":
        b, a = butter(filter_order, fband, btype='bandstop')
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
    
    # Apply filter to each column
    filtered_data = np.zeros_like(signal_data)
    
    if use_filtfilt:
        for jj in range(M):
            filtered_data[:, jj] = filtfilt(b, a, signal_data[:, jj])
    else:
        for jj in range(M):
            filtered_data[:, jj] = lfilter(b, a, signal_data[:, jj])
    
    return filtered_data


def yinterpolate(
    time_vector: npt.NDArray[np.floating],
    signal_data: npt.NDArray[np.floating],
    new_sample_rate: float
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Upsample signal to new sample rate with anti-aliasing filtering.
    
    Python port of MATLAB yinterpolate.m that upsamples signals using
    interpolation followed by anti-aliasing low-pass filtering.
    
    Args:
        time_vector: Original time vector, shape (N,)
        signal_data: Signal data, shape (N, M)
        new_sample_rate: New sample rate (must be higher than original)
        
    Returns:
        new_time: New time vector
        new_signal: Interpolated and filtered signal data
    """
    from scipy.interpolate import PchipInterpolator
    
    # Compute original sample rate
    Ts = np.mean(np.diff(time_vector))
    Fs = 1.0 / Ts
    
    if new_sample_rate <= Fs:
        raise ValueError("Only upsampling is supported")
    
    # Create new time vector
    Ts_new = 1.0 / new_sample_rate
    t_min, t_max = time_vector.min(), time_vector.max()
    new_time = np.arange(t_min, t_max + Ts_new, Ts_new)
    
    # Interpolate each channel
    N_new, M = len(new_time), signal_data.shape[1]
    new_signal = np.zeros((N_new, M))
    
    # Upsample factor and anti-aliasing cutoff
    r = new_sample_rate / Fs
    w_cut = 1.0 / r
    
    for cc in range(M):
        # Interpolate using PCHIP (similar to MATLAB's pchip)
        interpolator = PchipInterpolator(time_vector, signal_data[:, cc])
        new_signal[:, cc] = interpolator(new_time)
        
        # Apply anti-aliasing filter
        new_signal[:, cc] = yfilt(
            new_signal[:, cc:cc+1], 
            Ts=Ts_new,
            filter_type="LP", 
            frequency_band=w_cut,
            filter_order=8
        ).ravel()
    
    return new_time, new_signal


def qplot_data(
    data: npt.NDArray[np.floating],
    quantiles: Optional[npt.NDArray[np.floating]] = None,
    column_labels: Optional[List[str]] = None
) -> Tuple[npt.NDArray[np.floating], List[str]]:
    """
    Compute quantile statistics for plotting.
    
    Python port of MATLAB qplot.m that computes quantile statistics
    for each column of data matrix. Returns data for plotting rather
    than creating plots directly.
    
    Args:
        data: Data matrix, shape (n, m)
        quantiles: Quantile values to compute (default: [0.01, 0.1, 0.33, 0.5, 0.67, 0.9, 0.99])
        column_labels: Column labels (generated if None)
        
    Returns:
        quantile_values: Quantile values for each column, shape (7, m)
        labels: Column labels
    """
    if quantiles is None:
        quantiles = np.array([0.01, 0.1, 1/3, 0.5, 2/3, 0.9, 0.99])
    
    if len(quantiles) != 7:
        raise ValueError("quantiles must have 7 elements")
    
    if np.any(quantiles >= 1) or np.any(quantiles <= 0):
        raise ValueError("all quantile values must be in (0,1)")
    
    quantiles = np.sort(quantiles)
    
    n, m = data.shape
    
    # Generate labels if not provided
    if column_labels is None:
        column_labels = [f"#{i+1}" for i in range(m)]
    elif len(column_labels) != m:
        raise ValueError("Number of labels must match number of columns")
    
    # Compute quantiles for each column
    quantile_values = np.zeros((7, m))
    
    for ii in range(m):
        column_data = np.sort(data[:, ii])
        indices = np.round(quantiles * (n - 1)).astype(int)
        quantile_values[:, ii] = column_data[indices]
    
    return quantile_values, column_labels


# =============================================================================
# FFT Spectral Analysis Functions (Missing from MATLAB)
# =============================================================================

def fftspec(
    time_vector: Optional[NDArray[np.number]],
    signal_data: NDArray[np.number], 
    block_params: Tuple[int, int],
    nfft: Union[int, Tuple[int, int]],
    reduced_dim: int = 0,
    contour_levels: int = 25
) -> FFTSpectralResult:
    """
    Multi-channel FFT-based power spectral density analysis with optional random projection.
    
    This function computes block-based FFT spectral analysis of multivariate time series,
    with optional dimensionality reduction via random projection for computational efficiency.
    
    Args:
        time_vector: Time vector or None for sample indices
        signal_data: Input data matrix, shape (n_samples, n_channels)
        block_params: (block_size, block_stride) for windowing
        nfft: FFT length, or (nfft, smoothing_span) 
        reduced_dim: If > 0, project to this dimension before FFT
        contour_levels: Number of contour levels for visualization
        
    Returns:
        FFTSpectralResult containing power spectral density and frequency information
        
    Example:
        >>> t = np.linspace(0, 10, 1000)
        >>> y = np.sin(2*np.pi*t)[:, np.newaxis]
        >>> result = fftspec(t, y, (256, 128), 512)
        >>> print(f"PSD shape: {result.P.shape}")
    """
    if signal_data.ndim != 2:
        raise ValueError(f"signal_data must be 2D, got shape {signal_data.shape}")
    
    n_samples, n_channels = signal_data.shape
    block_size, block_stride = block_params
    
    # Handle nfft parameter
    if isinstance(nfft, (tuple, list)):
        nfft_len, n_smooth = nfft[:2]
    else:
        nfft_len, n_smooth = nfft, 1
    
    # Handle time vector
    if time_vector is None:
        time_vector = np.arange(n_samples)
        fs = 1.0
    else:
        time_vector = np.asarray(time_vector).flatten()
        if len(time_vector) != n_samples:
            raise ValueError("Length of time_vector does not match signal_data")
        fs = 1.0 / (time_vector[1] - time_vector[0]) if len(time_vector) > 1 else 1.0
    
    # Apply random projection if requested
    if reduced_dim > 0 and reduced_dim < n_channels:
        projection_matrix = np.random.randn(reduced_dim, n_channels)
        signal_data = (projection_matrix @ signal_data.T).T
        n_channels = reduced_dim
    
    # Compute block-based FFT
    block_starts = np.arange(0, n_samples - block_size + 1, block_stride)
    n_blocks = len(block_starts)
    
    # Initialize output arrays
    freq_bins = fftfreq(nfft_len, 1/fs)[:nfft_len//2]  # Positive frequencies only
    n_freq = len(freq_bins)
    psd_matrix = np.zeros((n_freq, n_blocks))
    block_times = np.zeros(n_blocks)
    
    # Process each block
    for i, start_idx in enumerate(block_starts):
        end_idx = start_idx + block_size
        block_data = signal_data[start_idx:end_idx, :]
        block_times[i] = time_vector[start_idx + block_size // 2]
        
        # Remove mean from each channel
        block_data = block_data - np.mean(block_data, axis=0)
        
        # Compute FFT for each channel and aggregate
        block_psd = np.zeros(n_freq)
        for ch in range(n_channels):
            # Zero-pad if necessary
            if block_size < nfft_len:
                padded_data = np.zeros(nfft_len)
                padded_data[:block_size] = block_data[:, ch]
            else:
                padded_data = block_data[:nfft_len, ch]
            
            # Compute FFT and PSD
            fft_data = fft(padded_data)
            channel_psd = np.abs(fft_data[:n_freq])**2 / (fs * nfft_len)
            block_psd += channel_psd
        
        # Average across channels
        psd_matrix[:, i] = block_psd / n_channels
    
    # Apply smoothing if requested
    if n_smooth > 1:
        from scipy.ndimage import uniform_filter1d
        psd_matrix = uniform_filter1d(psd_matrix, size=n_smooth, axis=0)
    
    return FFTSpectralResult(
        P=psd_matrix,
        F=freq_bins,
        T=block_times,
        nfft=nfft_len,
        block_size=block_size,
        n_blocks=n_blocks,
        fs=fs
    )


def fftspec1(
    signal: NDArray[np.number],
    nfft: int,
    overlap: float = 0.5,
    window: str = 'hann'
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Single-channel FFT spectral analysis using Welch's method.
    
    Args:
        signal: Single-channel time series
        nfft: FFT length
        overlap: Overlap fraction (0 to 1)
        window: Window function name
        
    Returns:
        Tuple of (frequencies, power spectral density)
    """
    from scipy.signal import welch
    
    if signal.ndim != 1:
        signal = signal.flatten()
    
    nperseg = nfft
    noverlap = int(nperseg * overlap)
    
    frequencies, psd = welch(
        signal, 
        fs=1.0,
        window=window, 
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft
    )
    
    return frequencies, psd


def fftspecwin(
    signal_data: NDArray[np.number],
    window_params: Dict[str, Union[str, int, float]]
) -> FFTSpectralResult:
    """
    Windowed FFT spectral analysis with various window functions.
    
    Args:
        signal_data: Input data matrix, shape (n_samples, n_channels)
        window_params: Dictionary with window parameters:
            - 'type': Window type ('hann', 'hamming', 'blackman', etc.)
            - 'nfft': FFT length
            - 'overlap': Overlap fraction
            - 'detrend': Detrending method ('linear', 'constant', None)
            
    Returns:
        FFTSpectralResult containing windowed spectral analysis
    """
    from scipy.signal import spectrogram
    
    window_type = window_params.get('type', 'hann')
    nfft = window_params.get('nfft', 1024)
    overlap = window_params.get('overlap', 0.5)
    detrend_method = window_params.get('detrend', 'constant')
    
    if signal_data.ndim == 1:
        signal_data = signal_data[:, np.newaxis]
    
    n_samples, n_channels = signal_data.shape
    noverlap = int(nfft * overlap)
    
    # Compute spectrogram for each channel and aggregate
    total_psd = None
    frequencies = None
    times = None
    
    for ch in range(n_channels):
        f, t, psd = spectrogram(
            signal_data[:, ch],
            fs=1.0,
            window=window_type,
            nperseg=nfft,
            noverlap=noverlap,
            nfft=nfft,
            detrend=detrend_method
        )
        
        if total_psd is None:
            total_psd = psd
            frequencies = f
            times = t
        else:
            total_psd += psd
    
    # Average across channels
    total_psd /= n_channels
    
    return FFTSpectralResult(
        P=total_psd,
        F=frequencies,
        T=times,
        nfft=nfft,
        block_size=nfft,
        n_blocks=len(times),
        fs=1.0
    )


def zmfftspec(
    signal_data: NDArray[np.number],
    remove_mean: bool = True,
    remove_trend: bool = False
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Zero-mean FFT spectral analysis.
    
    Args:
        signal_data: Input signal, shape (n_samples,) or (n_samples, n_channels)
        remove_mean: Whether to remove DC component
        remove_trend: Whether to remove linear trend
        
    Returns:
        Tuple of (frequencies, power spectral density)
    """
    if signal_data.ndim == 1:
        signal_data = signal_data[:, np.newaxis]
    
    n_samples, n_channels = signal_data.shape
    
    # Preprocess signal
    processed_data = signal_data.copy()
    
    if remove_trend:
        from scipy.signal import detrend
        for ch in range(n_channels):
            processed_data[:, ch] = detrend(processed_data[:, ch])
    elif remove_mean:
        processed_data = processed_data - np.mean(processed_data, axis=0)
    
    # Compute FFT
    fft_data = fft(processed_data, axis=0)
    frequencies = fftfreq(n_samples)[:n_samples//2]
    
    # Compute PSD and average across channels
    psd = np.mean(np.abs(fft_data[:n_samples//2, :])**2, axis=1) / n_samples
    
    return frequencies, psd


# =============================================================================
# Signal Conditioning Functions (Missing from MATLAB)
# =============================================================================

def yintegrate(
    signal: NDArray[np.number],
    method: str = 'trapz',
    initial_value: float = 0.0
) -> NDArray[np.floating]:
    """
    Numerical integration of time series using various methods.
    
    Args:
        signal: Input signal to integrate
        method: Integration method ('trapz', 'simpson', 'cumsum')
        initial_value: Initial condition for integration
        
    Returns:
        Integrated signal
    """
    if method == 'trapz':
        from scipy.integrate import cumulative_trapezoid
        return initial_value + cumulative_trapezoid(signal, initial=0)
    elif method == 'simpson':
        from scipy.integrate import simpson
        # For cumulative Simpson's rule, we need to compute incrementally
        integrated = np.zeros_like(signal)
        integrated[0] = initial_value
        for i in range(1, len(signal)):
            if i == 1:
                integrated[i] = integrated[i-1] + (signal[i] + signal[i-1]) / 2
            else:
                integrated[i] = integrated[i-1] + simpson(signal[i-1:i+1])
        return integrated
    elif method == 'cumsum':
        return initial_value + np.cumsum(signal)
    else:
        raise ValueError(f"Unknown integration method: {method}")


def ydecimate(
    signal: NDArray[np.number],
    decimation_factor: int,
    filter_order: int = 8,
    filter_type: str = 'iir'
) -> NDArray[np.floating]:
    """
    Decimation with anti-aliasing filtering.
    
    Args:
        signal: Input signal to decimate
        decimation_factor: Factor by which to reduce sampling rate
        filter_order: Order of anti-aliasing filter
        filter_type: Type of filter ('iir' or 'fir')
        
    Returns:
        Decimated signal
    """
    from scipy.signal import decimate
    
    return decimate(signal, decimation_factor, n=filter_order, ftype=filter_type)


def yresample(
    signal: NDArray[np.number],
    original_rate: float,
    target_rate: float,
    method: str = 'linear'
) -> NDArray[np.floating]:
    """
    Resample signal to new sampling rate with interpolation.
    
    Args:
        signal: Input signal to resample
        original_rate: Original sampling rate
        target_rate: Target sampling rate  
        method: Interpolation method ('linear', 'cubic', 'nearest')
        
    Returns:
        Resampled signal
    """
    from scipy.interpolate import interp1d
    
    n_original = len(signal)
    n_target = int(n_original * target_rate / original_rate)
    
    # Create time vectors
    t_original = np.linspace(0, n_original / original_rate, n_original)
    t_target = np.linspace(0, n_original / original_rate, n_target)
    
    # Interpolate
    interpolator = interp1d(t_original, signal, kind=method, 
                           bounds_error=False, fill_value='extrapolate')
    
    return interpolator(t_target)


def yaddgauss(
    signal: NDArray[np.number],
    noise_level: float,
    random_seed: Optional[int] = None
) -> NDArray[np.floating]:
    """
    Add Gaussian noise to signal.
    
    Args:
        signal: Input signal
        noise_level: Standard deviation of noise relative to signal
        random_seed: Random seed for reproducibility
        
    Returns:
        Signal with added Gaussian noise
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    signal_std = np.std(signal)
    noise = np.random.normal(0, noise_level * signal_std, signal.shape)
    
    return signal + noise


# =============================================================================
# Advanced Spectral Analysis Functions (Missing from MATLAB)
# =============================================================================

def fdmspec1(
    time_vector: Optional[NDArray[np.number]],
    signal_data: NDArray[np.number],
    block_params: Tuple[int, int],
    fdm_params: Tuple[int, int, int, float, int],
    threshold: Union[float, Tuple[float, float]] = 0.99,
    random_seed: Optional[int] = None
) -> Dict:
    """
    Block-based finite difference modal (FDM) frequency analysis.
    
    This function implements FDM-like frequency analysis using compressed sampling
    and random projections, following the MATLAB fdmspec1.m algorithm.
    
    Args:
        time_vector: Time vector or None for sample indices
        signal_data: Input data matrix, shape (n_samples, n_channels)
        block_params: (block_size, block_stride) for windowing
        fdm_params: (r, d1, d2, alpha, K) where:
            - r: Reduced dimension (<0 for orthonormal, >0 for random, 0 for none)
            - d1, d2: FDM eigenproblem sizes
            - alpha: Regularization parameter (<0 for auto, 0 for none)
            - K: Maximum number of frequencies to return
        threshold: Threshold value(s) close to 1 (e.g., 0.99)
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing FDM analysis results in eigspec-compatible format
        
    Example:
        >>> t = np.linspace(0, 10, 1000)
        >>> y = np.sin(2*np.pi*5*t)[:, np.newaxis]
        >>> result = fdmspec1(t, y, (256, 128), (-10, 50, 75, -1, 10))
    """
    if signal_data.ndim != 2:
        raise ValueError(f"signal_data must be 2D, got shape {signal_data.shape}")
    if len(block_params) != 2:
        raise ValueError("block_params must have 2 elements")
    if len(fdm_params) != 5:
        raise ValueError("fdm_params must have 5 elements")
    
    if isinstance(threshold, (int, float)):
        threshold = (threshold, threshold)
    elif len(threshold) != 2:
        raise ValueError("threshold must be scalar or 2-element tuple")
    
    if not all(0 <= t <= 1 for t in threshold):
        raise ValueError("threshold values must be between 0 and 1")
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n_samples, n_channels = signal_data.shape
    block_size, block_stride = block_params
    r, d1, d2, alpha, K = fdm_params
    
    # Handle time vector
    if time_vector is None:
        time_vector = np.arange(n_samples)
        time_step = -1
    else:
        time_vector = np.asarray(time_vector).flatten()
        if len(time_vector) != n_samples:
            raise ValueError("Length of time_vector does not match signal_data")
        time_step = time_vector[1] - time_vector[0] if len(time_vector) > 1 else 1.0
    
    # Calculate block parameters
    block_starts = np.arange(0, n_samples - block_size + 1, block_stride)
    n_blocks = len(block_starts)
    
    # Initialize results structure (mock-up to match rndspec format)
    block_results = []
    
    for i, start_idx in enumerate(block_starts):
        end_idx = start_idx + block_size
        block_data = signal_data[start_idx:end_idx, :].copy()
        
        # Demean block
        block_data = block_data - np.mean(block_data, axis=0)
        
        # Apply random projection if specified
        if r != 0:
            if r < 0:
                # Orthonormal projection
                projection_matrix = np.linalg.qr(np.random.randn(n_channels, -r))[0].T
                projected_data = (projection_matrix @ block_data.T).T
            else:
                # Random projection
                projection_matrix = np.random.randn(r, n_channels)
                projected_data = (projection_matrix @ block_data.T).T
        else:
            projection_matrix = None
            projected_data = block_data
        
        # FDM frequency analysis using eigenvalue-based method
        frequencies, amplitudes = _fdm_eigenanalysis(
            projected_data, d1, d2, alpha, K, threshold[0]
        )
        
        # Create mock modal report structure
        block_result = {
            'modal_report': {
                'frequencies': frequencies,
                'amplitudes': amplitudes,
                'lambda_vals': np.exp(1j * 2 * np.pi * frequencies) if len(frequencies) > 0 else np.array([]),
                'imode': list(range(len(frequencies))) if len(frequencies) > 0 else []
            },
            'shape_estimates': None,
            'projection_matrix': projection_matrix,
            'processing_time': 0.0,  # Would be filled in real implementation
            'demean_block': True,
            'time_step': time_step,
            'time_slice': (start_idx, end_idx),
            'centre_time': float((time_vector[start_idx] + time_vector[end_idx-1]) / 2),
            'filter_time': float(time_vector[end_idx-1])
        }
        
        block_results.append(block_result)
    
    return {
        'block_results': block_results,
        'fdm_params': fdm_params,
        'block_params': block_params,
        'threshold': threshold,
        'routine': 'fdmspec1'
    }


def _fdm_eigenanalysis(
    data: NDArray[np.number],
    d1: int,
    d2: int,
    alpha: float,
    max_frequencies: int,
    threshold: float
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Core FDM eigenanalysis for frequency extraction.
    
    This is a simplified implementation of the FDM eigenanalysis procedure.
    """
    n_samples, n_channels = data.shape
    
    if n_samples < max(d1, d2) + 1:
        return np.array([]), np.array([])
    
    # Build Hankel-like matrices for FDM
    # This is a simplified version - full implementation would be more complex
    
    # Use autocorrelation-based approach as approximation
    frequencies = []
    amplitudes = []
    
    # For each channel, find dominant frequencies using autocorrelation
    for ch in range(n_channels):
        signal = data[:, ch]
        
        # Compute autocorrelation
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find peaks in autocorrelation (simplified frequency detection)
        if len(autocorr) > 10:
            from scipy.signal import find_peaks
            peaks, properties = find_peaks(autocorr[1:], height=threshold * np.max(autocorr))
            
            # Convert peak positions to frequencies
            for peak in peaks[:max_frequencies//n_channels]:
                if peak > 0:
                    freq = 1.0 / (peak + 1)  # Simplified frequency estimation
                    amp = autocorr[peak + 1]
                    frequencies.append(freq)
                    amplitudes.append(amp)
    
    # Sort by amplitude and keep top frequencies
    if len(frequencies) > 0:
        freq_amp_pairs = sorted(zip(frequencies, amplitudes), key=lambda x: x[1], reverse=True)
        frequencies = np.array([f for f, a in freq_amp_pairs[:max_frequencies]])
        amplitudes = np.array([a for f, a in freq_amp_pairs[:max_frequencies]])
    else:
        frequencies = np.array([])
        amplitudes = np.array([])
    
    return frequencies, amplitudes


def kdftspec(
    time_vector: Optional[NDArray[np.number]],
    signal_data: NDArray[np.number],
    block_params: Tuple[int, int],
    dft_params: Union[int, Tuple[int, int]],
    reduced_dim: int = 0,
    k_folds: int = 8
) -> FFTSpectralResult:
    """
    K-fold DFT spectral analysis with robustified pruning of bursty data.
    
    This function implements k-fold cross-validated DFT analysis that acts as
    a de-ELMing filter for spectrograms, following MATLAB kdftspec.m.
    
    Args:
        time_vector: Time vector or None for sample indices
        signal_data: Input data matrix, shape (n_samples, n_channels)
        block_params: (block_size, block_stride) for windowing
        dft_params: DFT length, or (dft_length, smoothing_span)
        reduced_dim: Reduced dimension for random projection (0 for none)
        k_folds: Number of folds for robustification
        
    Returns:
        FFTSpectralResult with robustified spectral analysis
        
    Example:
        >>> t = np.linspace(0, 10, 1000)
        >>> y = np.sin(2*np.pi*t)[:, np.newaxis]
        >>> result = kdftspec(t, y, (512, 128), (2048, 3), 10, 8)
    """
    if signal_data.ndim != 2:
        raise ValueError(f"signal_data must be 2D, got shape {signal_data.shape}")
    if len(block_params) != 2:
        raise ValueError("block_params must have 2 elements")
    
    n_samples, n_channels = signal_data.shape
    block_size, block_stride = block_params
    
    # Handle DFT parameters
    if isinstance(dft_params, (int, float)):
        dft_length, smoothing_span = int(dft_params), 1
    else:
        dft_length, smoothing_span = dft_params[:2]
    
    # Handle time vector
    if time_vector is None:
        time_vector = np.arange(n_samples)
        fs = 1.0
    else:
        time_vector = np.asarray(time_vector).flatten()
        if len(time_vector) != n_samples:
            raise ValueError("Length of time_vector does not match signal_data")
        fs = 1.0 / (time_vector[1] - time_vector[0]) if len(time_vector) > 1 else 1.0
    
    # Apply random projection if specified
    if reduced_dim > 0 and reduced_dim < n_channels:
        projection_matrix = np.random.randn(reduced_dim, n_channels)
        signal_data = (projection_matrix @ signal_data.T).T
        n_channels = reduced_dim
    
    # Compute block-based DFT with k-fold robustification
    block_starts = np.arange(0, n_samples - block_size + 1, block_stride)
    n_blocks = len(block_starts)
    
    # Initialize output arrays
    freq_bins = fftfreq(dft_length, 1/fs)[:dft_length//2]
    n_freq = len(freq_bins)
    psd_matrix = np.zeros((n_freq, n_blocks))
    block_times = np.zeros(n_blocks)
    
    # Process each block with k-fold robustification
    for i, start_idx in enumerate(block_starts):
        end_idx = start_idx + block_size
        block_data = signal_data[start_idx:end_idx, :]
        block_times[i] = time_vector[start_idx + block_size // 2]
        
        # Remove mean from each channel
        block_data = block_data - np.mean(block_data, axis=0)
        
        # K-fold cross-validation for robustification
        fold_psds = []
        fold_size = block_size // k_folds
        
        for fold in range(k_folds):
            fold_start = fold * fold_size
            fold_end = min(fold_start + fold_size, block_size)
            
            if fold_end <= fold_start:
                continue
                
            fold_data = block_data[fold_start:fold_end, :]
            
            # Compute FFT for this fold
            fold_psd = np.zeros(n_freq)
            for ch in range(n_channels):
                # Zero-pad if necessary
                if len(fold_data) < dft_length:
                    padded_data = np.zeros(dft_length)
                    padded_data[:len(fold_data)] = fold_data[:, ch]
                else:
                    padded_data = fold_data[:dft_length, ch]
                
                # Compute FFT and PSD
                fft_data = fft(padded_data)
                channel_psd = np.abs(fft_data[:n_freq])**2 / (fs * dft_length)
                fold_psd += channel_psd
            
            fold_psds.append(fold_psd / n_channels)
        
        # Robustified PSD using median or mean of folds
        if len(fold_psds) > 0:
            fold_psds = np.array(fold_psds)
            # Use median for robustification (removes outliers/bursts)
            psd_matrix[:, i] = np.median(fold_psds, axis=0)
        else:
            psd_matrix[:, i] = 0
    
    # Apply smoothing if requested
    if smoothing_span > 1:
        if smoothing_span > 0:
            # Moving average smoothing
            from scipy.ndimage import uniform_filter1d
            psd_matrix = uniform_filter1d(psd_matrix, size=smoothing_span, axis=0)
        else:
            # Median filter smoothing
            from scipy.ndimage import median_filter
            psd_matrix = median_filter(psd_matrix, size=(-smoothing_span, 1))
    
    return FFTSpectralResult(
        P=psd_matrix,
        F=freq_bins,
        T=block_times,
        nfft=dft_length,
        block_size=block_size,
        n_blocks=n_blocks,
        fs=fs
    ) 