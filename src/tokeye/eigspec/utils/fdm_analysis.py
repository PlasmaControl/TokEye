"""
Finite Difference Method (FDM) analysis for eigspec package.

This module provides FDM-based frequency detection algorithms that are fundamental
to the eigspec toolbox:
- fdm1d: Core 1D FDM frequency detection algorithm
- Related FDM utilities and helper functions

Based on the MATLAB eigspec toolbox FDM implementations.
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np
import numpy.typing as npt
from scipy.linalg import eig, svd
import warnings


def fdm1d(
    Y: npt.NDArray[np.floating],
    d: int,
    alpha: float = -1.0
) -> Dict[str, Any]:
    """
    FDM-type frequency detection for multichannel time-series data.
    
    Python port of MATLAB fdm1d.m that implements finite difference method
    for finding frequency lists from multichannel time-series. Uses generalized
    eigenvalue problems with regularization to detect dominant frequencies.
    
    Args:
        Y: Signal data matrix where columns are channels, shape (n, m)
        d: Desired eigenvalue problem dimension
        alpha: Regularization parameter (negative for auto-selection)
        
    Returns:
        Dictionary containing:
        - eigk: Sorted eigenvalues (decreasing plausibility)
        - errk: Error estimates for each eigenvalue
        - tallness: Data matrix tallness ratio
        - p: Snapshot length
        - alpha: Used regularization parameter
    """
    default_epscut = 1e-12
    minimum_snapshot = 10
    
    n, m = Y.shape
    p = n - d
    
    if p < minimum_snapshot:
        raise ValueError("Snapshot length is too small")
    
    # Check tallness requirement
    tallness = m * p / (n - p + 1)
    if tallness < 2:
        raise ValueError("Code requires data matrix to be significantly tall")
    
    # Build data matrix D
    nd = n - p + 1
    D = np.zeros((m * p, nd))
    
    for tt in range(p, n):  # MATLAB: (p+1):(n+1), but 0-indexed
        jj = tt - p
        # Extract p consecutive samples in reverse order for each channel
        block = Y[tt:tt-p:-1, :].T.ravel()  # Equivalent to MATLAB reshape
        D[:, jj] = block
    
    # Construct shifted matrices for generalized eigenvalue problem
    S = D[:, :-1]  # D(:,1:(nd-1))
    R = D[:, 1:]   # D(:,2:nd)
    
    nm = S.shape[1]
    
    # Auto-select regularization parameter if needed
    if alpha < 0:
        # Based on eigenvalues of S'*S following Werner&Cary approach
        StS = S.T @ S
        eig_vals = np.linalg.eigvals(StS)
        eig_vals = np.sort(np.real(eig_vals))[::-1]  # Descending order
        
        # Use median of eigenvalues as regularization (heuristic)
        alpha = float(np.median(eig_vals)) * 1e-6
        
        if alpha < default_epscut:
            alpha = default_epscut
    
    # Solve regularized generalized eigenvalue problem
    # (R'*R + alpha*I) * v = lambda * (S'*S + alpha*I) * v
    A = R.T @ R + alpha * np.eye(nm)
    B = S.T @ S + alpha * np.eye(nm)
    
    try:
        eigenvals, eigenvecs = eig(A, B)
    except np.linalg.LinAlgError as e:
        warnings.warn(f"Eigenvalue computation failed: {e}")
        return {
            'eigk': np.array([]),
            'errk': np.array([]),
            'tallness': tallness,
            'p': p,
            'alpha': alpha
        }
    
    # Convert to complex eigenvalues (frequencies)
    frequencies = np.log(eigenvals) / (1j * 2 * np.pi)
    
    # Compute error estimates (based on residual norms)
    error_estimates = np.zeros(len(frequencies))
    
    for i, (eigval, eigvec) in enumerate(zip(eigenvals, eigenvecs.T)):
        if np.abs(eigval) > default_epscut:
            # Compute residual for error estimate
            residual_A = A @ eigvec - eigval * (B @ eigvec)
            residual_norm = np.linalg.norm(residual_A)
            
            # Normalize by eigenvalue magnitude
            error_estimates[i] = residual_norm / max(np.abs(eigval), default_epscut)
        else:
            error_estimates[i] = np.inf
    
    # Sort by error estimates (ascending = most plausible first)
    sort_idx = np.argsort(error_estimates)
    sorted_frequencies = frequencies[sort_idx]
    sorted_errors = error_estimates[sort_idx]
    
    # Convert error estimates to confidence values (higher = better)
    confidence = 1.0 / (1.0 + sorted_errors)
    
    return {
        'eigk': sorted_frequencies,
        'errk': confidence,
        'tallness': tallness,
        'p': p,
        'alpha': alpha,
        'raw_eigenvals': eigenvals[sort_idx],
        'eigenvecs': eigenvecs[:, sort_idx]
    }


def rndspec(
    time_vector: Optional[npt.NDArray[np.floating]],
    signal_data: npt.NDArray[np.floating],
    block_params: Tuple[int, int],
    analysis_params: Tuple[int, ...],
    thresholds: Tuple[float, float],
    use_cca: bool = False
) -> Dict[str, Any]:
    """
    Block-based compressed sampling SSI/AR/PCA analysis.
    
    Python port of MATLAB rndspec.m that performs block-based spectral analysis
    using random projection and various identification methods.
    
    Args:
        time_vector: Time vector or None for sample indices
        signal_data: Signal data matrix, shape (N, M)
        block_params: [block_size, block_stride]
        analysis_params: [reduced_dim, future, past, order1, order2] for SSI
                        or [reduced_dim, past, order1, order2] for AR/PCA
        thresholds: [MAC_threshold, DST_threshold]
        use_cca: Whether to use CCA/CVA in SSI subprogram
        
    Returns:
        Dictionary with analysis results for each block
    """
    block_size, block_stride = block_params
    N, M = signal_data.shape
    
    # Handle time vector
    if time_vector is None or (len(time_vector) == 1 and time_vector[0] == 0):
        time_vector = np.arange(1, N + 1)
        Ts = -1
    else:
        time_vector = time_vector.ravel()
        if len(time_vector) != N:
            raise ValueError("Length of time vector does not match signal data")
        Ts = time_vector[1] - time_vector[0]
    
    # Block analysis setup
    t1_vec = np.arange(0, N - block_size + 1, block_stride)
    n_blocks = len(t1_vec)
    
    # Initialize results
    block_results = []
    
    for jj in range(n_blocks):
        t1 = t1_vec[jj]
        t2 = t1 + block_size
        
        # Extract block data
        block_data = signal_data[t1:t2, :]
        
        # Process block using appropriate method
        if len(analysis_params) == 5:
            # SSI mode
            block_result = _process_ssi_block(
                block_data, analysis_params, thresholds, use_cca, Ts
            )
        else:
            # AR/PCA mode
            block_result = _process_arpca_block(
                block_data, analysis_params, thresholds, Ts
            )
        
        # Add timing and block info
        block_result['t1t2'] = [t1, t2]
        block_result['centre_t'] = (time_vector[t1] + time_vector[t2-1]) / 2
        block_result['filter_t'] = time_vector[t2-1]
        block_result['Ts'] = Ts
        
        block_results.append(block_result)
    
    return {
        'L': block_results,
        'bss': block_params,
        'rfpn': analysis_params,
        'thresh': thresholds,
        'routine': 'rndspec'
    }


def pcaspecx(
    time_vector: Optional[npt.NDArray[np.floating]],
    signal_data: npt.NDArray[np.floating],
    block_params: Tuple[int, int],
    analysis_params: Tuple[int, int, int, int, int],
    thresholds: Tuple[float, float],
    k_folds: int = 8
) -> Dict[str, Any]:
    """
    Block-based PCA-projected SSI analysis with k-fold pruning.
    
    Python port of MATLAB pcaspecx.m that uses PCA projection with k-fold
    pruning to reduce outlier effects on the projection.
    
    Args:
        time_vector: Time vector or None for sample indices
        signal_data: Signal data matrix, shape (N, M) 
        block_params: [block_size, block_stride]
        analysis_params: [reduced_dim, future, past, order1, order2]
        thresholds: [MAC_threshold, DST_threshold]
        k_folds: Number of folds for pruning (default 8)
        
    Returns:
        Dictionary with analysis results for each block
    """
    block_size, block_stride = block_params
    N, M = signal_data.shape
    
    # Handle time vector
    if time_vector is None or (len(time_vector) == 1 and time_vector[0] == 0):
        time_vector = np.arange(1, N + 1)
        Ts = -1
    else:
        time_vector = time_vector.ravel()
        if len(time_vector) != N:
            raise ValueError("Length of time vector does not match signal data")
        Ts = time_vector[1] - time_vector[0]
    
    # Block analysis setup
    t1_vec = np.arange(0, N - block_size + 1, block_stride)
    n_blocks = len(t1_vec)
    
    # Initialize results
    block_results = []
    
    for jj in range(n_blocks):
        t1 = t1_vec[jj]
        t2 = t1 + block_size
        
        # Extract block data
        block_data = signal_data[t1:t2, :]
        
        # Process block with PCA projection and k-fold pruning
        block_result = _process_pca_block(
            block_data, analysis_params, thresholds, k_folds, 'mcd'
        )
        
        # Add timing and block info
        block_result['t1t2'] = [t1, t2]
        block_result['centre_t'] = (time_vector[t1] + time_vector[t2-1]) / 2
        block_result['filter_t'] = time_vector[t2-1]
        block_result['Ts'] = Ts
        
        block_results.append(block_result)
    
    return {
        'L': block_results,
        'bss': block_params,
        'rfpn': analysis_params,
        'thresh': thresholds,
        'k': k_folds,
        'routine': 'pcaspecx'
    }


# Helper functions for block processing

def _process_ssi_block(
    block_data: npt.NDArray[np.floating],
    params: Tuple[int, int, int, int, int],
    thresholds: Tuple[float, float],
    use_cca: bool,
    Ts: float
) -> Dict[str, Any]:
    """Process a single block using SSI method."""
    # Placeholder implementation - would need full SSI algorithm
    from ..subspace_identification import covariance_driven_ssi
    
    reduced_dim, future, past, order1, order2 = params
    
    # Apply random projection if needed
    if reduced_dim > 0 and reduced_dim < block_data.shape[1]:
        projection = np.random.randn(reduced_dim, block_data.shape[1])
        if reduced_dim < 0:  # Orthonormalize
            projection = np.linalg.qr(projection.T)[0].T
        block_data = (projection @ block_data.T).T
    
    # Run SSI
    try:
        ssi_result = covariance_driven_ssi(
            block_data, 
            model_order=max(order1, order2),
            block_rows=future + past
        )
        
        # Extract modes and format results
        result = {
            'mrep': {
                'imode': np.arange(len(ssi_result.natural_frequencies)),
                'm0': {
                    'lambda': ssi_result.discrete_eigenvalues,
                    'shape': ssi_result.mode_shapes
                }
            },
            'method': 'ssi',
            'Psi': projection if reduced_dim > 0 else np.eye(block_data.shape[1])
        }
        
    except Exception as e:
        # Return empty result if SSI fails
        result = {
            'mrep': {'imode': np.array([]), 'm0': {'lambda': np.array([]), 'shape': np.array([])}},
            'method': 'ssi_failed',
            'error': str(e)
        }
    
    return result


def _process_arpca_block(
    block_data: npt.NDArray[np.floating],
    params: Tuple[int, int, int, int],
    thresholds: Tuple[float, float],
    Ts: float
) -> Dict[str, Any]:
    """Process a single block using AR/PCA method."""
    from ..autoregressive_pca import arpca
    
    reduced_dim, past, order1, order2 = params
    
    # Apply random projection if needed
    projection = np.eye(block_data.shape[1])
    if reduced_dim > 0 and reduced_dim < block_data.shape[1]:
        projection = np.random.randn(reduced_dim, block_data.shape[1])
        if reduced_dim < 0:  # Orthonormalize
            projection = np.linalg.qr(projection.T)[0].T
        block_data = (projection @ block_data.T).T
    
    # Run AR/PCA
    try:
        arpca_result = arpca(block_data, max(order1, order2), reduced_dim or block_data.shape[1])
        
        result = {
            'mrep': {
                'imode': np.arange(len(arpca_result.model.eigenvalues)),
                'm0': {
                    'lambda': arpca_result.model.eigenvalues,
                    'shape': arpca_result.model.mode_shapes
                }
            },
            'method': 'arpca',
            'Psi': projection
        }
        
    except Exception as e:
        result = {
            'mrep': {'imode': np.array([]), 'm0': {'lambda': np.array([]), 'shape': np.array([])}},
            'method': 'arpca_failed',
            'error': str(e)
        }
    
    return result


def _process_pca_block(
    block_data: npt.NDArray[np.floating],
    params: Tuple[int, int, int, int, int],
    thresholds: Tuple[float, float],
    k_folds: int,
    criterion: str
) -> Dict[str, Any]:
    """Process a single block using PCA projection with k-fold pruning."""
    from ..matlab_utilities import kfoldcov
    
    reduced_dim, future, past, order1, order2 = params
    
    # Apply k-fold pruning for robust PCA
    try:
        idx, mu, R, _ = kfoldcov(block_data.T, k_folds, criterion)
        pruned_data = block_data[idx, :]
        
        # Compute PCA projection
        if reduced_dim > 0 and reduced_dim < pruned_data.shape[1]:
            U, s, Vt = svd(R, full_matrices=False)
            projection = Vt[:reduced_dim, :]
            projected_data = (projection @ pruned_data.T).T
        else:
            projection = np.eye(pruned_data.shape[1])
            projected_data = pruned_data
        
        # Process with SSI
        result = _process_ssi_block(projected_data, params, thresholds, False, 1.0)
        result['Psi'] = projection
        result['pruned_indices'] = idx
        result['method'] = 'pcaspecx'
        
    except Exception as e:
        result = {
            'mrep': {'imode': np.array([]), 'm0': {'lambda': np.array([]), 'shape': np.array([])}},
            'method': 'pcaspecx_failed',
            'error': str(e)
        }
    
    return result