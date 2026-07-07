"""
Data extraction and collection utilities for eigspec package.

This module provides functions for extracting and organizing analysis results:
- extract_ptrefs: Extract reference points from analysis results
- collect_rep_data: Harvest features from analysis results 
- Data organization and formatting utilities

Based on the MATLAB eigspec toolbox data collection functions.
"""

from typing import Optional, List, Tuple, Dict, Any, Union
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass


@dataclass
class PointReference:
    """Single point reference extracted from analysis results."""
    query_tifr: Tuple[float, float]  # [time_ms, freq_kHz]
    shapevector: npt.NDArray[np.complex128]
    rms: float
    frequency: float  # Hz
    radius: float
    centre_time: float  # seconds


@dataclass  
class PointReferences:
    """Collection of point references with metadata."""
    modes: List[PointReference]
    sensor_coordinates: npt.NDArray[np.floating]
    Ts: float  # sampling period in seconds
    block_params: Tuple[int, int]
    analysis_params: Tuple[int, ...]
    thresholds: Tuple[float, float]


def extract_ptrefs(
    analysis_result: Any,
    query_points: npt.NDArray[np.floating],
    sensor_coordinates: npt.NDArray[np.floating],
    warning_distance: float = 5.0
) -> PointReferences:
    """
    Extract reference points from analysis results.
    
    Python port of MATLAB extract_ptrefs.m that finds the closest modal
    features to provided query points in (time, frequency) space.
    
    Args:
        analysis_result: Analysis result structure with .L blocks
        query_points: Query points [time_ms, freq_kHz], shape (n_refs, 2)  
        sensor_coordinates: Sensor coordinate matrix
        warning_distance: Distance threshold for warnings
        
    Returns:
        PointReferences object with extracted reference points
    """
    if not hasattr(analysis_result, 'L') or not analysis_result.L:
        raise ValueError("Analysis result must have non-empty L field")
    
    n_refs = query_points.shape[0]
    if query_points.shape[1] != 2:
        raise ValueError("Query points must have shape (n_refs, 2) for [time, freq]")
    
    # Collect all available features
    XTRF, XRMS, XS = collect_rep_data(analysis_result)
    
    if XTRF.shape[0] == 0:
        raise ValueError("No modal features found in analysis results")
    
    # Convert frequency to kHz for comparison
    XTRF_search = XTRF.copy()
    XTRF_search[:, 2] = XTRF_search[:, 2] / 1e3  # Convert Hz to kHz
    
    # Extract metadata
    first_block = analysis_result.L[0]
    Ts = getattr(first_block, 'Ts', -1)
    block_params = getattr(analysis_result, 'bss', (0, 0))
    analysis_params = getattr(analysis_result, 'rfpn', ())
    thresholds = getattr(analysis_result, 'thresh', (0.0, 0.0))
    
    # Find closest matches for each query point
    extracted_modes = []
    
    for jj in range(n_refs):
        query_time, query_freq = query_points[jj]
        
        # Find closest point in (time, frequency) space
        distances_sq = (
            (XTRF_search[:, 0] - query_time)**2 + 
            (XTRF_search[:, 2] - query_freq)**2
        )
        
        closest_idx = np.argmin(distances_sq)
        distance = np.sqrt(distances_sq[closest_idx])
        
        closest_time = XTRF_search[closest_idx, 0]
        closest_freq = XTRF_search[closest_idx, 2]
        
        print(f"Query {jj+1}: closest to [time,freq]=[{query_time:.6f},{query_freq:.6f}] "
              f"found at index {closest_idx+1}/{len(XTRF)} "
              f"[time,freq]=[{closest_time:.6f},{closest_freq:.6f}] (ms,kHz)")
        
        if distance > warning_distance:
            print(f"(Warning: distance to query point is large, d={distance:.3f})")
        
        # Create point reference
        mode_ref = PointReference(
            query_tifr=(query_time, query_freq),
            shapevector=XS[:, closest_idx],
            rms=XRMS[closest_idx],
            frequency=XTRF[closest_idx, 2],  # Hz
            radius=XTRF[closest_idx, 1],
            centre_time=XTRF[closest_idx, 0] / 1e3  # Convert ms to seconds
        )
        
        extracted_modes.append(mode_ref)
    
    return PointReferences(
        modes=extracted_modes,
        sensor_coordinates=sensor_coordinates,
        Ts=Ts,
        block_params=block_params,
        analysis_params=analysis_params,
        thresholds=thresholds
    )


def collect_rep_data(analysis_result: Any) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.complex128]]:
    """
    Harvest features from analysis results and store in arrays.
    
    Python port of MATLAB collect_rep_data.m that extracts all modal
    features from block-based analysis results.
    
    Args:
        analysis_result: Analysis result structure with .L blocks
        
    Returns:
        XTRF: Feature matrix [time_ms, damping, frequency_Hz], shape (n_modes, 3)
        XRMS: RMS values for each mode, shape (n_modes,)
        XS: Complex mode shape vectors, shape (n_channels, n_modes)
    """
    if not hasattr(analysis_result, 'L') or not analysis_result.L:
        return np.array([]).reshape(0, 3), np.array([]), np.array([]).reshape(0, 0)
    
    # Determine number of channels
    first_block = analysis_result.L[0]
    if hasattr(analysis_result, 'iisubset'):
        M = len(analysis_result.iisubset)
    elif hasattr(first_block, 'Psi'):
        M = first_block.Psi.shape[1] if hasattr(first_block.Psi, 'shape') else 1
    else:
        M = 1  # Default fallback
    
    Ts = getattr(first_block, 'Ts', -1)
    n_blocks = len(analysis_result.L)
    
    # Count total modes and collect block times
    T = np.zeros(n_blocks)
    total_modes = 0
    
    for jj, block in enumerate(analysis_result.L):
        if hasattr(block, 'mrep') and hasattr(block.mrep, 'imode'):
            total_modes += len(block.mrep.imode)
        
        if hasattr(block, 'centre_t'):
            T[jj] = block.centre_t
        else:
            T[jj] = jj  # Default time indexing
    
    # Initialize output arrays
    if total_modes == 0:
        return np.array([]).reshape(0, 3), np.array([]), np.array([]).reshape(0, 0)
    
    XTRF = np.zeros((total_modes, 3))  # [time_ms, damping, frequency_Hz]
    XRMS = np.zeros(total_modes)
    XS = np.zeros((M, total_modes), dtype=np.complex128)
    
    # Extract features from each block
    mode_idx = 0
    
    for jj, block in enumerate(analysis_result.L):
        if not (hasattr(block, 'mrep') and hasattr(block.mrep, 'imode')):
            continue
            
        n_modes_block = len(block.mrep.imode)
        
        for mm in range(n_modes_block):
            # Get modal shape vector
            C = _get_c_shape(block, mm, M)
            
            # Get frequency and damping
            radius, frequency = _get_freq(block, mm, Ts)
            
            # Store data
            XTRF[mode_idx, :] = [1e3 * T[jj], radius, frequency]  # time in ms, freq in Hz
            XS[:, mode_idx] = C
            
            # Get RMS if available
            if (hasattr(block, 'drep') and hasattr(block.drep, 'drms') and 
                mm < len(block.drep.drms)):
                XRMS[mode_idx] = block.drep.drms[mm]
            else:
                XRMS[mode_idx] = np.linalg.norm(C)  # Default to vector norm
            
            mode_idx += 1
    
    return XTRF, XRMS, XS


def _get_freq(block: Any, mode_idx: int, Ts: float) -> Tuple[float, float]:
    """Extract frequency and damping from block data."""
    if not (hasattr(block, 'mrep') and hasattr(block.mrep, 'm0') and 
            hasattr(block.mrep.m0, 'lambda')):
        return 0.0, 0.0
    
    if mode_idx >= len(block.mrep.imode):
        return 0.0, 0.0
    
    mode_number = block.mrep.imode[mode_idx]
    # 'lambda' is a keyword: attribute access must go through getattr
    eigenvalues = getattr(block.mrep.m0, 'lambda')
    if mode_number >= len(eigenvalues):
        return 0.0, 0.0

    eigenvalue = eigenvalues[mode_number]
    
    # Extract radius (damping) and frequency
    radius = np.abs(eigenvalue)
    frequency = np.angle(eigenvalue)
    
    # Convert to Hz if sampling period is provided
    if Ts > 0:
        frequency = frequency / (2 * np.pi * Ts)
    
    return float(radius), float(frequency)


def _get_c_shape(block: Any, mode_idx: int, n_channels: int) -> npt.NDArray[np.complex128]:
    """Extract complex shape vector from block data."""
    if not (hasattr(block, 'mrep') and hasattr(block.mrep, 'm0')):
        return np.zeros(n_channels, dtype=np.complex128)
    
    if mode_idx >= len(block.mrep.imode):
        return np.zeros(n_channels, dtype=np.complex128)
    
    # Try to get shape from various possible locations
    if hasattr(block.mrep.m0, 'shape'):
        shapes = block.mrep.m0.shape
        if shapes.size > 0:
            if shapes.ndim == 2 and mode_idx < shapes.shape[1]:
                return shapes[:, mode_idx]
            elif shapes.ndim == 1 and mode_idx == 0:
                return shapes
    
    # Try alternative shape storage
    if hasattr(block, 'drep') and hasattr(block.drep, 'dhat'):
        dhat = block.drep.dhat
        if dhat.size > 0:
            # Reconstruct complex shape from real/imaginary parts
            if dhat.ndim == 2 and mode_idx * 2 + 1 < dhat.shape[1]:
                real_part = dhat[:, mode_idx * 2]
                imag_part = dhat[:, mode_idx * 2 + 1]
                return real_part + 1j * imag_part
    
    # Default fallback
    return np.zeros(n_channels, dtype=np.complex128)


def collect_prototype_traces(
    analysis_result: Any,
    point_references: PointReferences,
    time_vector: npt.NDArray[np.floating],
    signal_data: npt.NDArray[np.floating]
) -> Dict[str, npt.NDArray[np.floating]]:
    """
    Collect prototype time traces for reference points.
    
    Args:
        analysis_result: Analysis result structure
        point_references: Reference points extracted from analysis
        time_vector: Time vector for signal data
        signal_data: Original signal data matrix
        
    Returns:
        Dictionary with prototype traces for each reference point
    """
    n_refs = len(point_references.modes)
    prototype_traces = {}
    
    for i, mode_ref in enumerate(point_references.modes):
        # Find time window around reference point
        center_time = mode_ref.centre_time
        frequency = mode_ref.frequency
        
        # Use several periods for prototype extraction
        if frequency > 0:
            period = 1.0 / frequency
            window_duration = min(5 * period, 0.1)  # 5 periods or 100ms max
        else:
            window_duration = 0.05  # 50ms default
        
        # Find indices for time window
        time_mask = (np.abs(time_vector - center_time) <= window_duration / 2)
        
        if np.any(time_mask):
            window_data = signal_data[time_mask, :]
            window_time = time_vector[time_mask] - center_time  # Relative time
            
            # Compute weighted projection using shape vector
            shape_vec = mode_ref.shapevector
            if len(shape_vec) == window_data.shape[1]:
                # Project signal onto mode shape
                prototype = np.real(window_data @ np.conj(shape_vec))
                
                prototype_traces[f'mode_{i+1}'] = {
                    'time': window_time,
                    'signal': prototype,
                    'reference_info': {
                        'frequency': frequency,
                        'rms': mode_ref.rms,
                        'query_point': mode_ref.query_tifr
                    }
                }
    
    return prototype_traces


def format_analysis_summary(analysis_result: Any) -> Dict[str, Any]:
    """
    Create a summary of analysis results.
    
    Args:
        analysis_result: Analysis result structure
        
    Returns:
        Dictionary with analysis summary statistics
    """
    summary = {
        'n_blocks': 0,
        'total_modes': 0,
        'frequency_range': [0.0, 0.0],
        'time_range': [0.0, 0.0],
        'processing_method': 'unknown'
    }
    
    if not hasattr(analysis_result, 'L') or not analysis_result.L:
        return summary
    
    # Basic counts
    summary['n_blocks'] = len(analysis_result.L)
    summary['processing_method'] = getattr(analysis_result, 'routine', 'unknown')
    
    # Collect features for statistics
    XTRF, XRMS, XS = collect_rep_data(analysis_result)
    
    if XTRF.shape[0] > 0:
        summary['total_modes'] = XTRF.shape[0]
        summary['time_range'] = [float(XTRF[:, 0].min()), float(XTRF[:, 0].max())]
        summary['frequency_range'] = [float(XTRF[:, 2].min()), float(XTRF[:, 2].max())]
        summary['rms_range'] = [float(XRMS.min()), float(XRMS.max())]
        summary['damping_range'] = [float(XTRF[:, 1].min()), float(XTRF[:, 1].max())]
    
    # Add parameter information if available
    if hasattr(analysis_result, 'bss'):
        summary['block_parameters'] = analysis_result.bss
    if hasattr(analysis_result, 'rfpn'):
        summary['analysis_parameters'] = analysis_result.rfpn
    if hasattr(analysis_result, 'thresh'):
        summary['thresholds'] = analysis_result.thresh
    
    return summary