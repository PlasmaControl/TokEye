"""
Random projection spectral analysis for eigspec package.

This module provides the main random projection spectral analysis functionality:
- High-level interface for multi-channel time series analysis
- Automated modal identification with random dimensionality reduction
- Comprehensive analysis workflow combining SSI, AR-PCA, and modal analysis

Based on the MATLAB eigspec toolbox main analysis functions:
- rndspecx.m - Core random projection spectral analysis algorithm
- eigspec_mmain.m - Main analysis entry point and workflow
- view_pcaspec_results.m - Results processing and classification
- view_pcaspec_prototypes.m - Prototype-based modal analysis
"""

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ..utils.block_processing import random_projection_block_analysis
from ..utils.utils import demean

@dataclass
class RandomProjectionBlockResult:
    """
    Container for individual block processing results in spectral analysis.
    
    Attributes:
        reduced_dimension_matrix: Modal analysis results
        reduced_dimension_array: Shape estimates  
        projection_matrix: Random projection matrix used for this block
        processing_time: Time taken to process block in seconds
        demean_block: Whether block was demeaned before processing
        time_step: Time step between samples in seconds
        time_slice: (start_idx, end_idx) time indices of block
        centre_time: Center time of block in seconds
        filter_time: Filter time in seconds
    """
    reduced_dimension_matrix: object  # ModalShortlist
    reduced_dimension_array: object   # ShapeEstimates or None
    projection_matrix: Optional[NDArray[np.floating]]  # Random projection matrix
    processing_time: float
    demean_block: bool
    time_step: float
    time_slice: Tuple[int, int]
    centre_time: float
    filter_time: float
    
    # Add property aliases for backward compatibility with tests
    @property
    def modal_report(self) -> object:
        """Alias for reduced_dimension_matrix for test compatibility."""
        return self.reduced_dimension_matrix

@dataclass
class RandomProjectionSpectralAnalysisResult:
    """
    Container for spectral analysis results.
    
    Attributes:
        block_results: List of results for each processed block
        total_processing_time: Total processing time in seconds
        reduced_dimension: Analysis parameters:
            For SSI: [reduced_dim, future, past, order1, order2]
            For AR/PCA: [reduced_dim, past, order1, order2]
            where:
            - reduced_dim: Target dimension (<0 for orthonormal, >0 for random, 0 for no projection)
            - future/past: Number of future/past samples for SSI
            - order1/order2: Model orders
        block_parameters: (block_size, block_stride) for block processing
        threshold_parameters: (MAC threshold, DST threshold) for mode matching
        use_canonical_correlation_analysis: Whether CCA/CVA was used in SSI
        analysis_name: Name of analysis routine used
    """
    block_results: List[RandomProjectionBlockResult]
    total_processing_time: float
    reduced_dimension: List[int]
    block_parameters: Tuple[int, int]
    threshold_parameters: Tuple[float, float]
    use_canonical_correlation_analysis: bool
    analysis_name: str

def random_projection_spectral_analysis(
    time_array: Union[None, NDArray[np.number], float, int],
    signal_array: NDArray[np.number],
    block_parameters: Tuple[int, int],
    reduced_dimension: List[int],
    threshold_parameters: Tuple[float, float],
    use_canonical_correlation_analysis: bool = False,
    random_seed: Optional[int] = None
) -> RandomProjectionSpectralAnalysisResult:
    """
    Block-based spectral analysis of multivariate time-series using random projections.
    
    Args:
        time_array: Time vector, scalar 0, or None. If None or 0, uses sample indices.
        signal_array: Input data matrix, shape (n_samples, n_channels)
        block_parameters: (block_size, block_stride) for block processing
        reduced_dimension: Analysis parameters:
            For SSI: [reduced_dim, future, past, order1, order2]
            For AR/PCA: [reduced_dim, past, order1, order2]
            where:
            - reduced_dim: Target dimension (<0 for orthonormal, >0 for random, 0 for no projection)
            - future/past: Number of future/past samples for SSI
            - order1/order2: Model orders
        threshold_parameters: (MAC threshold, DST threshold) for mode matching
        use_canonical_correlation_analysis: Whether to use CCA/CVA in SSI
        random_seed: Optional seed for random number generation
        
    Returns:
        RandomProjectionSpectralAnalysisResult containing analysis results
        
    Example:
        >>> t = np.linspace(0, 10, 1000)
        >>> y = np.sin(2*np.pi*t)[:, np.newaxis]
        >>> block_params = (100, 50)  # 100-sample blocks with 50-sample stride
        >>> reduced_dim = [-5, 10, 10, 2, 2]  # SSI with orthonormal projection to 5D
        >>> thresh = (0.9, 0.9)  # MAC and distance thresholds
        >>> result = random_projection_spectral_analysis(t, y, block_params, reduced_dim, thresh)
    """
    if signal_array.ndim != 2:
        raise ValueError(f"signal_array must be 2D, got shape {signal_array.shape}")
    if not isinstance(reduced_dimension, list) or not all(isinstance(x, int) for x in reduced_dimension):
        raise TypeError("reduced_dimension must be a list of integers")
    if len(reduced_dimension) not in (4, 5):
        raise ValueError("reduced_dimension must have 4 elements (AR/PCA) or 5 elements (SSI)")
        
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
        
    n, _ = signal_array.shape
    if time_array is None or (isinstance(time_array, (int, float)) and time_array == 0):
        time_vector = np.arange(n)
        time_step = -1
    else:
        time_vector = np.asarray(time_array).reshape(-1)
        if len(time_vector) != n:
            raise ValueError("Length of time_array does not match length of signal_array")
        time_step = time_vector[1] - time_vector[0]
        
    block_size, block_stride = block_parameters
    block_start_indices = np.arange(0, n + 1 - block_size, block_stride)
    n_block = len(block_start_indices)
    
    # Initialize results
    block_results: List[RandomProjectionBlockResult] = []
    
    ttl = time.time()
    
    for block_index in range(n_block):
        time_start = block_start_indices[block_index]
        time_end = time_start + block_size - 1
        
        # Process block, time-slice [time_start, time_end]
        start_time = time.time()
        
        # Demean and process block
        demeaned_block = demean(signal_array[time_start:time_end+1, :])
        random_projection_result = random_projection_block_analysis(
            demeaned_block, reduced_dimension, threshold_parameters, 
            use_canonical_correlation_analysis, random_seed
        )
        
        # Store results
        # Process all the channels in one go - if can parallelize, then don't need to do random projection
        block_result = RandomProjectionBlockResult(
            reduced_dimension_matrix=random_projection_result.modal_analysis,
            reduced_dimension_array=random_projection_result.shape_estimates,
            projection_matrix=random_projection_result.projection_matrix,
            processing_time=time.time() - start_time,
            demean_block=True,
            time_step=time_step,
            time_slice=(time_start, time_end),
            centre_time=float((time_vector[time_start] + time_vector[time_end]) / 2),
            filter_time=float(time_vector[time_end])
        )
        block_results.append(block_result)
        
    # Create final results
    result = RandomProjectionSpectralAnalysisResult(
        block_results=block_results,
        total_processing_time=time.time() - ttl,
        reduced_dimension=reduced_dimension,
        block_parameters=block_parameters,
        threshold_parameters=threshold_parameters,
        use_canonical_correlation_analysis=use_canonical_correlation_analysis,
        analysis_name=f"random_projection_spectral_analysis({'ssi' if len(reduced_dimension) == 5 else 'ar/pca'})"
    )
    
    return result 
