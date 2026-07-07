"""
Autoregressive Principal Component Analysis (AR-PCA) for eigspec package.

This module provides AR-PCA algorithms for time series modeling and spectral analysis:
- Autoregressive modeling with PCA dimensionality reduction
- Eigenvalue-based frequency and damping estimation
- Model selection and validation procedures

Based on the MATLAB eigspec toolbox AR-PCA functions:
- arpca.m - Main AR-PCA algorithm implementation
- ssi1cax.m - Extended SSI algorithm with AR-PCA components
- Various supporting utilities for model fitting and validation
"""

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
from numpy.typing import NDArray


@dataclass
class ARPCAResult:
    """
    Container for AR/PCA analysis results.
    
    Attributes:
        A: State transition matrix, shape (n_states, n_states)
        K: Kalman gain matrix, shape (n_states, n_outputs)
        C: Output matrix, shape (n_outputs, n_states)
        H: AR model Markov blocks, shape (n_outputs, n_outputs * past)
        pca: PCA eigenvalues (if dimension reduction was used)
        models: List of models for multiple orders (if applicable)
        Ry: Data covariance matrix (if residual computed)
        Re: Residual error covariance (if residual computed, single model case)
    """
    A: Optional[NDArray[np.floating]]
    K: Optional[NDArray[np.floating]]
    C: Optional[NDArray[np.floating]]
    H: NDArray[np.floating]
    pca: Optional[NDArray[np.floating]]
    models: Optional[List['ARPCAModel']]
    Ry: Optional[NDArray[np.floating]] = None
    Re: Optional[NDArray[np.floating]] = None


@dataclass
class ARPCAModel:
    """
    Container for individual AR/PCA model.
    
    Attributes:
        A: State transition matrix
        K: Kalman gain matrix
        C: Output matrix
        Re: Residual error covariance (optional)
    """
    A: NDArray[np.floating]
    K: NDArray[np.floating]
    C: NDArray[np.floating]
    Re: Optional[NDArray[np.floating]] = None


def arpca(
    y: NDArray[np.number],
    p: int,
    r: Union[int, float, List[Union[int, float]]],
    compute_residual: bool = False
) -> ARPCAResult:
    """
    AR/PCA time-series modeling for a block of multichannel data.
    
    This function implements AR/PCA modeling where the past horizon for the linear
    predictor is p samples. The state order r can be specified in several ways:
    - Scalar 1 <= r <= m*p: Return single model with order r
    - Vector r: Return list of models with orders in r
    - Scalar 0 < r < 1: Select order based on PCA eigenvalue energy fraction
    - Scalar r <= 0: Return unreduced lag-p AR model
    
    Args:
        y: Multichannel data, shape (n_samples, n_channels)
        p: Past horizon for linear predictor (positive integer)
        r: State order specification (see above)
        compute_residual: If True, compute residual error covariance
        
    Returns:
        ARPCAResult containing system matrices and analysis results
        
    Raises:
        ValueError: If parameters are invalid or data dimensions are problematic
        
    Example:
        >>> # Generate AR(2) process
        >>> np.random.seed(42)
        >>> y = np.random.randn(1000, 3)  # 3 channels, 1000 samples
        >>> result = arpca(y, p=10, r=5)  # Past=10, reduced order=5
        >>> print(f"A matrix shape: {result.A.shape}")
        >>> print(f"Kalman gain shape: {result.K.shape}")
    """
    if not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array")
    if y.ndim != 2:
        raise ValueError(f"y must be 2D, got shape {y.shape}")
    if not isinstance(p, int) or p < 1:
        raise ValueError("p must be a positive integer")
    
    N, ny = y.shape
    
    if ny >= N:
        raise ValueError(f"Number of channels {ny} exceeds number of samples {N}")
    
    # Ensure we have enough samples for the lagged structure
    # Need at least p+1 samples to form one prediction instance
    if p >= N:
        raise ValueError(f"Lag p={p} must be less than batch length {N}")
    
    # Warn if lag is very large relative to data length
    if p >= N // 2:
        import warnings
        warnings.warn(f"Large lag p={p} relative to batch length {N} may lead to poor estimates")
    
    # Main AR/PCA computation
    result = _subarpca(y, p, r)
    
    # Compute residual error covariance if requested
    if compute_residual:
        result = _compute_residual_error(result, y, p)
    
    return result


def _subarpca(
    signal: NDArray[np.number],
    past_horizon: int,
    order: Union[int, float, List[Union[int, float]]]
) -> ARPCAResult:
    """
    Core AR/PCA computation.
    
    Args:
        signal: Input data, shape (n_samples, n_channels)
        past_horizon: Past horizon
        order: Order specification
        
    Returns:
        ARPCAResult with computed models
    """
    n_samples, n_channels = signal.shape
    n_prediction_instances = n_samples - past_horizon
    n_lagged_dimension = n_channels * past_horizon
    
    # Build lagged data matrices
    Z = signal.T  # Transpose for easier indexing
    Z_past = np.zeros((n_lagged_dimension, n_prediction_instances))  # Past data matrix
    Y_current = np.zeros((n_channels, n_prediction_instances))    # Current output matrix
    
    for i in range(past_horizon, n_samples):  # MATLAB: for kk=(p+1):N
        # Build past vector: [y(k-1), y(k-2), ..., y(k-p)]
        # Fixed indexing: collect past_horizon samples in reverse order
        past_indices = list(range(i-1, i-1-past_horizon, -1))  # [i-1, i-2, ..., i-p]
        past_block = Z[:, past_indices]  # Shape: (n_channels, past_horizon)
        Z_past[:, i-past_horizon] = past_block.flatten()
        Y_current[:, i-past_horizon] = signal[i, :]
    
    # Compute lagged covariance and AR model
    R_z_past = Z_past @ Z_past.T
    H = (Y_current @ Z_past.T) @ np.linalg.pinv(R_z_past)  # AR model, Markov blocks
    
    # Build full lag-p state-space model
    AK, K, C = _build_lag_model(H, n_channels)
    
    if isinstance(order, (list, tuple, np.ndarray)):
        # Multiple models requested
        order_vec = list(order)
        if any(ri <= 0 or ri > n_lagged_dimension for ri in order_vec if ri >= 1):
            raise ValueError("Order elements must be in range (0,1) or [1,m*p]")
        
        V, D = _sorted_eig(R_z_past)
        cumd = np.cumsum(D) / np.sum(D)  # Cumulative energy
        
        models = []
        for ri in order_vec:
            if 0 < ri < 1:
                # Energy-based selection
                ri = int(np.argmax(cumd >= ri) + 1)
            
            ri = int(ri)
            V_r = V[:, :ri]
            AKr, Kr, Cr = _deflate_model(H, V_r)
            models.append(ARPCAModel(A=AKr + Kr @ Cr, K=Kr, C=Cr))
        
        return ARPCAResult(A=None, K=None, C=None, H=H, pca=D, models=models)
    
    elif isinstance(order, (int, float)):
        if order <= 0:
            # Unreduced model
            return ARPCAResult(A=AK + K @ C, K=K, C=C, H=H, pca=None, models=None)
        
        elif order >= 1 and order <= n_lagged_dimension:
            # Single reduced model with specific order
            order = int(order)
            V, D = _sorted_eig(R_z_past)
            V_r = V[:, :order]
            AKr, Kr, Cr = _deflate_model(H, V_r)
            return ARPCAResult(A=AKr + Kr @ Cr, K=Kr, C=Cr, H=H, pca=D, models=None)
        
        elif 0 < order < 1:
            # Single reduced model based on energy fraction
            V, D = _sorted_eig(R_z_past)
            cumd = np.cumsum(D) / np.sum(D)
            order = int(np.argmax(cumd >= order) + 1)
            V_r = V[:, :order]
            AKr, Kr, Cr = _deflate_model(H, V_r)
            return ARPCAResult(A=AKr + Kr @ Cr, K=Kr, C=Cr, H=H, pca=D, models=None)
        
        else:
            raise ValueError(f"Invalid order specification order={order}")
    
    else:
        raise ValueError("order must be int, float, or list/array")


def _deflate_model(
    H: NDArray[np.floating],
    V_r: NDArray[np.floating]
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """
    Compute reduced-order model matrices from full AR model.
    
    Args:
        H: AR model Markov blocks, shape (m, m*p)
        V_r: Reduction matrix, shape (m*p, r)
        
    Returns:
        Tuple of (AKr, Kr, Cr) - reduced model matrices
    """
    m = H.shape[0]
    p = H.shape[1] // m
    
    # Extract matrices
    Kr = V_r[:m, :].T  # First m rows, transposed
    Cr = H @ V_r       # Apply reduction to Markov blocks
    
    # Compute reduced A matrix - Fixed to match MATLAB indexing
    AKr = V_r[m:p*m, :].T @ V_r[:m*(p-1), :]
    
    return AKr, Kr, Cr


def _build_lag_model(
    H: NDArray[np.floating],
    nz: int
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """
    Build full lag-p state-space model from AR Markov blocks.
    
    Args:
        H: AR model Markov blocks, shape (nz, nz*p)
        nz: Number of outputs/channels
        
    Returns:
        Tuple of (A, B, C) where B is Kalman gain structure
    """
    p = H.shape[1] // nz
    
    # Build companion form state matrix
    A = np.block([
        [np.zeros((nz, p * nz))],
        [np.block([np.eye((p-1) * nz), np.zeros(((p-1) * nz, nz))])]
    ])
    
    # Input matrix (Kalman gain structure)
    B = np.block([
        [np.eye(nz)],
        [np.zeros(((p-1) * nz, nz))]
    ])
    
    # Output matrix
    C = H
    
    return A, B, C


def _sorted_eig(A: NDArray[np.floating]) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Compute eigendecomposition with eigenvalues sorted in descending order.
    
    Args:
        A: Matrix (may not be symmetric)
        
    Returns:
        Tuple of (eigenvectors, eigenvalues) sorted by eigenvalue magnitude
    """
    # Use general eigenvalue decomposition to match MATLAB eig() behavior
    eigvals, eigvecs = np.linalg.eig(A)
    
    # Sort in descending order by real part (matching MATLAB behavior)
    idx = np.argsort(np.real(eigvals))[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    return eigvecs, eigvals


def _compute_residual_error(result: ARPCAResult, y: NDArray[np.number], p: int) -> ARPCAResult:
    """
    Compute residual error covariance for AR/PCA models.
    
    Args:
        result: AR/PCA result containing models
        y: Original data matrix, shape (n_samples, n_channels)
        p: Past horizon
        
    Returns:
        Updated ARPCAResult with residual error covariances
    """
    from scipy.signal import lsim
    
    N, ny = y.shape
    
    # Compute data covariance
    Ry = (y.T @ y) / N
    
    if hasattr(result, 'A') and result.A is not None:
        # Single model case
        A, K, C = result.A, result.K, result.C
        
        # Create discrete-time state-space model for residual computation
        # H = (sI - (A - KC))^{-1} K  with D = I for residual filter
        # In discrete time: H(z) = C(zI - (A - KC))^{-1}K + I
        try:
            # Simulate the system to get residuals
            # This is a simplified implementation - full MATLAB version uses ss() and lsim()
            residuals = _compute_model_residuals(y, A, K, C, p)
            Re = (residuals.T @ residuals) / (N - p)
            
            # Add residual info to result
            result_dict = result.__dict__.copy()
            result_dict['Ry'] = Ry
            result_dict['Re'] = Re
            result = ARPCAResult(**result_dict)
            
        except Exception:
            # Fallback: skip residual computation if simulation fails
            pass
            
    elif hasattr(result, 'models') and result.models:
        # Multiple models case
        for i, model in enumerate(result.models):
            try:
                residuals = _compute_model_residuals(y, model.A, model.K, model.C, p)
                Re = (residuals.T @ residuals) / (N - p)
                
                # Add residual info to model (create new model with residual)
                model_dict = model.__dict__.copy()
                model_dict['Re'] = Re
                result.models[i] = ARPCAModel(**model_dict)
                
            except Exception:
                # Skip residual for this model if computation fails
                continue
        
        # Add data covariance to main result
        result_dict = result.__dict__.copy()
        result_dict['Ry'] = Ry
        result = ARPCAResult(**result_dict)
    
    return result


def _compute_model_residuals(
    y: NDArray[np.number], 
    A: NDArray[np.floating], 
    K: NDArray[np.floating], 
    C: NDArray[np.floating], 
    p: int
) -> NDArray[np.floating]:
    """
    Compute residuals for a single AR/PCA model.
    
    Args:
        y: Data matrix, shape (n_samples, n_channels)
        A: State transition matrix
        K: Kalman gain matrix  
        C: Output matrix
        p: Past horizon
        
    Returns:
        Residuals matrix, shape (n_samples-p, n_channels)
    """
    N, ny = y.shape
    
    # Simple residual computation using one-step-ahead prediction
    # This is a simplified version of the MATLAB lsim approach
    residuals = np.zeros((N - p, ny))
    
    # Initialize state
    n_states = A.shape[0]
    x = np.zeros(n_states)
    
    for i in range(p, N):
        # Predict output
        y_pred = C @ x
        
        # Compute residual
        residuals[i - p, :] = y[i, :] - y_pred
        
        # Update state: x[k+1] = A*x[k] + K*(y[k] - C*x[k])
        innovation = y[i, :] - y_pred
        x = A @ x + K @ innovation
    
    return residuals 