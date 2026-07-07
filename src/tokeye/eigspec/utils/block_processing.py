"""
Block-based data processing for eigspec package.

This module provides block-based analysis functions for large dataset processing:
- Random projection block analysis with modal identification
- Time-windowed spectral analysis and feature extraction
- Block-based system identification workflows

Based on the MATLAB eigspec toolbox block processing functions:
- rndspecx.m - Random projection spectral analysis for individual blocks
- eigspec_mmain.m - Main block-wise analysis workflow
- collect_rep_data.m - Feature collection across time blocks
- view_pcaspec_results.m - Block-based results visualization
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .modal_analysis import ModalShortlist, ShapeEstimates, order_mac
from .subspace_identification import covariance_driven_ssi, canonical_correlation_ssi
from .autoregressive_pca import arpca

@dataclass
class BlockAnalysisResult:
    """
    Container for block processing results.
    
    Attributes:
        modal_analysis: Modal analysis results after dimension reduction
        shape_estimates: Shape estimates after dimension reduction
        processing_time: Time taken to process block in seconds
        demean_applied: Whether block was demeaned before processing
        time_step: Time step between samples in seconds
        time_indices: (start_idx, end_idx) time indices of block
        center_time: Center time of block in seconds
        filter_time: Filter time in seconds
    """
    modal_analysis: ModalShortlist
    shape_estimates: Optional[ShapeEstimates]
    processing_time: float
    demean_applied: bool
    time_step: float
    time_indices: Tuple[int, int]
    center_time: float
    filter_time: float

@dataclass
class RandomProjectionResult:
    """
    Container for random projection block results.
    
    Attributes:
        projection_matrix: Random projection matrix, shape (reduced_dim, n_channels) or None
        modal_analysis: Modal analysis results after projection
        shape_estimates: Shape estimates after projection
    """
    projection_matrix: Optional[NDArray[np.floating]]
    modal_analysis: ModalShortlist
    shape_estimates: Optional[ShapeEstimates]

def random_projection_block_analysis(
    data_block: NDArray[np.number],
    analysis_parameters: List[int],
    matching_thresholds: Tuple[float, float],
    use_canonical_correlation: bool = False,
    random_seed: Optional[int] = None
) -> RandomProjectionResult:
    """
    Process a data block using random projection and modal analysis.
    
    Args:
        data_block: Input data block, shape (n_samples, n_channels)
        analysis_parameters: Analysis configuration:
            For SSI: [reduced_dim, future_horizon, past_horizon, order1, order2]
            For AR/PCA: [reduced_dim, past_horizon, order1, order2]
            where:
            - reduced_dim: Target dimension (<0 for orthonormal, >0 for random, 0 for no projection)
            - future_horizon/past_horizon: Number of future/past samples for SSI
            - order1/order2: Model orders
        matching_thresholds: (MAC threshold, distance threshold) for mode matching
        use_canonical_correlation: Whether to use CCA/CVA in SSI algorithm
        random_seed: Optional seed for random number generation
        
    Returns:
        RandomProjectionResult containing:
        - Random projection matrix (if used)
        - Modal analysis results
        - Shape estimates
        
    Raises:
        ValueError: If reduced dimension is larger than number of channels
        
    Example:
        >>> data = np.random.randn(1000, 10)  # 1000 samples, 10 channels
        >>> params = [-5, 10, 10, 2, 2]  # SSI with orthonormal projection to 5D
        >>> thresholds = (0.9, 0.9)  # MAC and distance thresholds
        >>> result = random_projection_block_analysis(data, params, thresholds)
    """
    if data_block.ndim != 2:
        raise ValueError(f"data_block must be 2D, got shape {data_block.shape}")
    if not isinstance(analysis_parameters, list) or not all(isinstance(x, int) for x in analysis_parameters):
        raise TypeError("analysis_parameters must be a list of integers")
    if len(analysis_parameters) not in (4, 5):
        raise ValueError("analysis_parameters must have 4 elements (AR/PCA) or 5 elements (SSI)")
        
    n_samples, n_channels = data_block.shape
    reduced_dimension = analysis_parameters[0]
    
    if abs(reduced_dimension) > n_channels:
        raise ValueError(f"Reduced dimension {abs(reduced_dimension)} cannot exceed input dimension {n_channels}")
    
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    # TODO: Matrix with Random Variance. Orthonormal Projection with QR Factorization. Or Simpler way. 2 ways to do.
    # See if returns Q or R matrix. Want Q matrix.
    # Generate random projection matrix
    if reduced_dimension < 0:
        # Orthonormal projection
        projection_matrix = np.linalg.qr(np.random.randn(n_channels, -reduced_dimension))[0].T
    elif reduced_dimension > 0:
        # Random projection - matching MATLAB behavior (no normalization)
        projection_matrix = np.random.randn(reduced_dimension, n_channels)
    else:
        projection_matrix = None
    
    reduced_dimension = abs(reduced_dimension)

    # Apply projection if needed
    projected_data = (projection_matrix @ data_block.T).T if projection_matrix is not None else data_block

    # Make this its own section
    if len(analysis_parameters) == 4:
        # AR/PCA mode: [reduced_dim, past_horizon, order1, order2]
        past_horizon = analysis_parameters[1]
        model_orders = analysis_parameters[2:4]
        
        # Call ARPCA algorithm  
        arpca_result = arpca(projected_data, past_horizon, list(model_orders))
        
        if arpca_result.models:
            # Multiple models case
            first_model = arpca_result.models[0]
            second_model = arpca_result.models[1] if len(arpca_result.models) > 1 else first_model
            
            modal_analysis = order_mac(
                first_model.C, first_model.A, 
                second_model.C, second_model.A, 
                matching_thresholds
            )
        else:
            # Single model case - create dummy second model for OMAC comparison
            state_matrix, output_matrix = arpca_result.A, arpca_result.C
            if state_matrix is not None and output_matrix is not None:
                # Small perturbation for comparison
                perturbed_state = state_matrix * 1.01
                perturbed_output = output_matrix * 1.01
            else:
                raise ValueError("ARPCA returned None matrices")
            modal_analysis = order_mac(
                output_matrix, state_matrix, 
                perturbed_output, perturbed_state, 
                matching_thresholds
            )
            
    else:
        # SSI mode: [reduced_dim, future_horizon, past_horizon, order1, order2]
        future_horizon = analysis_parameters[1]
        past_horizon = analysis_parameters[2]
        model_orders = analysis_parameters[3:5]
        identification_params = [future_horizon, past_horizon] + model_orders
        
        if reduced_dimension == 0:
            # Full-signal analysis
            if use_canonical_correlation:
                # More computationally heavy
                ssi_result = canonical_correlation_ssi(projected_data, identification_params)
            else:
                # Use this by default
                ssi_result = covariance_driven_ssi(projected_data, identification_params)
        else:
            # Reduced-signal analysis (same as full signal since data is already projected)
            if use_canonical_correlation:
                ssi_result = canonical_correlation_ssi(projected_data, identification_params)
            else:
                ssi_result = covariance_driven_ssi(projected_data, identification_params)
        
        if ssi_result.models and len(ssi_result.models) >= 2:
            # Multiple models case
            first_model = ssi_result.models[0]
            second_model = ssi_result.models[1]
            modal_analysis = order_mac(
                first_model.output_matrix, first_model.state_matrix,
                second_model.output_matrix, second_model.state_matrix, 
                matching_thresholds
            )
        elif ssi_result.models and len(ssi_result.models) == 1:
            # Single model case - create dummy second model
            first_model = ssi_result.models[0]
            perturbed_state = first_model.state_matrix * 1.01
            perturbed_output = first_model.output_matrix * 1.01
            modal_analysis = order_mac(
                first_model.output_matrix, first_model.state_matrix,
                perturbed_output, perturbed_state, 
                matching_thresholds
            )
        else:
            # Use single model results directly
            state_matrix, output_matrix = ssi_result.state_matrix, ssi_result.output_matrix
            if state_matrix is not None and output_matrix is not None:
                perturbed_state, perturbed_output = state_matrix * 1.01, output_matrix * 1.01
                modal_analysis = order_mac(
                    output_matrix, state_matrix, 
                    perturbed_output, perturbed_state, 
                    matching_thresholds
                )
            else:
                raise ValueError("SSI returned None matrices")
    
    # Estimate shape vectors from original (unprojected) data
    from .modal_analysis import shapes_from_freq
    shape_estimates = shapes_from_freq(data_block, modal_analysis)
    
    return RandomProjectionResult(
        projection_matrix=projection_matrix, 
        modal_analysis=modal_analysis, 
        shape_estimates=shape_estimates
    ) 
