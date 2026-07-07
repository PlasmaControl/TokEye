"""
Subspace identification algorithms for eigspec package.

This module provides stochastic subspace identification (SSI) algorithms for system identification:
- Covariance-driven SSI (SSI-COV) with block Hankel matrices
- Canonical correlation analysis SSI (SSI-CCA) for robust identification
- State-space model extraction and validation

Based on the MATLAB eigspec toolbox subspace identification functions:
- ssi1ca.m - Covariance-driven stochastic subspace identification
- ssicca.m - Canonical correlation analysis subspace identification  
- ssi1cax.m - Extended SSI with cross-validation
- kfoldcov.m - K-fold cross-validation for covariance estimation
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray


@dataclass
class SubspaceIdentificationResult:
    """
    Container for subspace identification analysis results.
    
    Attributes:
        state_matrix: State transition matrix A, shape (n_states, n_states)
        output_matrix: Output matrix C, shape (n_outputs, n_states)
        singular_values: Singular values from SVD decomposition
        models: List of models for multiple orders (if applicable)
    """
    state_matrix: Optional[NDArray[np.floating]]
    output_matrix: Optional[NDArray[np.floating]]
    singular_values: NDArray[np.floating]
    models: Optional[List['StateSpaceModel']]


@dataclass 
class StateSpaceModel:
    """
    Container for individual state-space model.
    
    Attributes:
        state_matrix: State transition matrix A
        output_matrix: Output matrix C  
        kalman_gain: Kalman gain matrix K (for CCA variant)
    """
    state_matrix: NDArray[np.floating]
    output_matrix: NDArray[np.floating]
    kalman_gain: Optional[NDArray[np.floating]] = None


def covariance_driven_ssi(
    data: NDArray[np.number],
    identification_params: List[int]
) -> SubspaceIdentificationResult:
    """
    Covariance-driven stochastic subspace identification using classical SVD approach.
    
    This function implements the classical covariance-driven SSI algorithm that extracts
    state-space models (A, C) from multichannel time series data based on the extended
    observability matrix.
    
    Args:
        data: Multichannel data where each column is a channel time-series, 
              shape (n_samples, n_channels)
        identification_params: Analysis parameters [future, past, order1, order2, ...]
                             - future: Number of future samples in block-Hankel matrix
                             - past: Number of past samples in block-Hankel matrix  
                             - order1, order2, ...: System orders to estimate
             
    Returns:
        SubspaceIdentificationResult containing:
        - state_matrix, output_matrix for single order case
        - List of models for multiple orders
        - Singular values from SVD
        
    Raises:
        ValueError: If system order exceeds theoretical limit or parameters are invalid
        
    Example:
        >>> # Generate synthetic 2-channel oscillatory data
        >>> t = np.linspace(0, 10, 1000)
        >>> data = np.column_stack([np.sin(2*np.pi*t), np.cos(2*np.pi*t)])
        >>> params = [10, 10, 2, 4]  # future=10, past=10, orders=[2,4]
        >>> result = covariance_driven_ssi(data, params)
        >>> print(f"Found {len(result.models)} models")
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a numpy array")
    if data.ndim != 2:
        raise ValueError(f"data must be 2D, got shape {data.shape}")
    if not isinstance(identification_params, list) or len(identification_params) < 3:
        raise ValueError("identification_params must be a list with at least 3 elements [future, past, order1, ...]")
    
    n_samples, n_channels = data.shape
    future_horizon = identification_params[0]
    past_horizon = identification_params[1]  
    system_orders = identification_params[2:]
    
    if len(system_orders) == 0:
        raise ValueError("System order list is empty")
    if any(order > n_channels * future_horizon for order in system_orders):
        max_order = max(system_orders)
        raise ValueError(f"Order request {max_order} exceeds theoretical limit n_channels*future = {n_channels*future_horizon}")
    if future_horizon <= 0 or past_horizon <= 0:
        raise ValueError("Future and past horizons must be positive integers")
    if n_samples <= past_horizon + future_horizon:
        raise ValueError(f"Data length {n_samples} too short for past+future = {past_horizon+future_horizon}")
    
    # Construct block-Hankel data matrix
    n_data_columns = n_samples - past_horizon - future_horizon + 1
    n_data_rows = n_channels * (past_horizon + future_horizon)
    hankel_matrix = np.zeros((n_data_rows, n_data_columns))
    
    for col_idx in range(n_data_columns):
        # Extract data block from time col_idx to col_idx+past+future-1.
        # Rows must be time-block-major ([all channels at t0, all channels
        # at t1, ...]) to match the n_channels-strided slicing below (and
        # the MATLAB original); flattening the (time, channel) block in C
        # order gives exactly that.
        data_block = data[col_idx:col_idx+past_horizon+future_horizon, :]
        hankel_matrix[:, col_idx] = data_block.flatten()
    
    # Split into past and future components
    past_data = hankel_matrix[:n_channels*past_horizon, :]
    future_data = hankel_matrix[n_channels*past_horizon:, :]
    
    # Calculate the past-to-future projection matrix
    cross_covariance = (future_data @ past_data.T) / n_data_columns
    
    # SVD decomposition
    # Visualize SVD Output?
    left_singular_vectors, singular_values, right_singular_vectors_T = np.linalg.svd(cross_covariance, full_matrices=False)
    
    if len(system_orders) == 1:
        # Single model case
        order = system_orders[0]
        
        # Construct extended observability matrix
        observability_matrix = left_singular_vectors[:, :order] @ np.diag(np.sqrt(singular_values[:order]))
        
        # Extract system matrices
        output_matrix = observability_matrix[:n_channels, :]
        # MATLAB: A = O(1:(m*(f-1)),:) \ O((m+1):(m*f),:).
        # lstsq(O1, O2)[0] is exactly pinv(O1) @ O2 — the same as backslash;
        # no transpose is needed (or correct) here.
        state_matrix = np.linalg.lstsq(
            observability_matrix[:n_channels*(future_horizon-1), :],
            observability_matrix[n_channels:n_channels*future_horizon, :],
            rcond=None
        )[0]

        return SubspaceIdentificationResult(
            state_matrix=state_matrix, 
            output_matrix=output_matrix, 
            singular_values=singular_values, 
            models=None
        )
    
    else:
        # Multiple model case
        models = []
        for order in system_orders:
            # Construct extended observability matrix for this order
            observability_matrix = left_singular_vectors[:, :order] @ np.diag(np.sqrt(singular_values[:order]))
            
            # Extract system matrices
            output_matrix = observability_matrix[:n_channels, :]
            # Same as the single-model case: lstsq already matches backslash.
            state_matrix = np.linalg.lstsq(
                observability_matrix[:n_channels*(future_horizon-1), :],
                observability_matrix[n_channels:n_channels*future_horizon, :],
                rcond=None
            )[0]
            
            models.append(StateSpaceModel(state_matrix=state_matrix, output_matrix=output_matrix))
            
        return SubspaceIdentificationResult(
            state_matrix=None, 
            output_matrix=None, 
            singular_values=singular_values, 
            models=models
        )


def canonical_correlation_ssi(
    data: NDArray[np.number],
    identification_params: List[int],
    compute_residual: bool = False
) -> SubspaceIdentificationResult:
    """
    Stochastic subspace identification using canonical correlation analysis.
    
    This function implements the CCA-based SSI algorithm which can provide better
    numerical conditioning compared to basic covariance-driven SSI.
    
    Args:
        data: Multichannel data, shape (n_samples, n_channels)
        identification_params: Analysis parameters [future, past, order1, order2, ...]
        compute_residual: If True, compute residual error covariance
        
    Returns:
        SubspaceIdentificationResult containing models with A, K, C matrices and singular values
        
    Example:
        >>> data = np.random.randn(1000, 3)  # 3-channel data
        >>> params = [15, 15, 3]  # future=15, past=15, order=3
        >>> result = canonical_correlation_ssi(data, params)
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a numpy array")
    if data.ndim != 2:
        raise ValueError(f"data must be 2D, got shape {data.shape}")
    
    n_samples, n_channels = data.shape
    future_horizon = identification_params[0]
    past_horizon = identification_params[1]
    system_orders = identification_params[2:]
    
    if len(system_orders) == 0:
        raise ValueError("System order list is empty")
    if any(order > n_channels * future_horizon for order in system_orders):
        max_order = max(system_orders)
        raise ValueError(f"Order request exceeds theoretical limit n_channels*future = {n_channels*future_horizon}")
    
    # Construct block-Hankel data matrix
    n_data_columns = n_samples - past_horizon - future_horizon + 1
    n_data_rows = n_channels * (past_horizon + future_horizon)
    hankel_matrix = np.zeros((n_data_rows, n_data_columns))
    
    for col_idx in range(n_data_columns):
        data_block = data[col_idx:col_idx+past_horizon+future_horizon, :].T
        hankel_matrix[:, col_idx] = data_block.flatten()
    
    # Split into past and future components
    past_data = hankel_matrix[:n_channels*past_horizon, :]
    future_data = hankel_matrix[n_channels*past_horizon:, :]
    
    # Calculate covariance matrices for CCA
    future_covariance = (future_data @ future_data.T) / n_data_columns
    past_covariance = (past_data @ past_data.T) / n_data_columns  
    cross_covariance = (future_data @ past_data.T) / n_data_columns
    
    # Compute CCA weights using matrix square roots
    U1, S1, Vh1 = np.linalg.svd(future_covariance)
    inverse_sqrt_future_cov = Vh1.T @ np.diag(1.0 / np.sqrt(S1)) @ Vh1
    
    U1, S1, Vh1 = np.linalg.svd(past_covariance)
    inverse_sqrt_past_cov = Vh1.T @ np.diag(1.0 / np.sqrt(S1)) @ Vh1
    
    # CCA-weighted matrix
    cca_matrix = inverse_sqrt_future_cov @ cross_covariance @ inverse_sqrt_past_cov
    left_vectors, singular_values, right_vectors_T = np.linalg.svd(cca_matrix, full_matrices=False)
    
    if len(system_orders) == 1:
        # Single model case
        order = system_orders[0]
        
        # Compute state sequence
        state_sequence = (right_vectors_T[:order, :] @ inverse_sqrt_past_cov) @ past_data
        output_sequence = data[past_horizon:n_samples-future_horizon+1, :].T
        
        # Estimate output matrix C
        output_matrix = output_sequence @ np.linalg.pinv(state_sequence)
        
        # Estimate A and K matrices
        n_time_steps = state_sequence.shape[1]
        augmented_regression_matrix = np.vstack([
            state_sequence[:, :n_time_steps-1], 
            data[past_horizon:n_samples-future_horizon, :].T
        ])
        next_states = state_sequence[:, 1:n_time_steps]
        
        system_kalman_matrix = next_states @ np.linalg.pinv(augmented_regression_matrix)
        state_matrix = system_kalman_matrix[:, :order]
        kalman_gain = system_kalman_matrix[:, order:order+n_channels]
        final_state_matrix = state_matrix + kalman_gain @ output_matrix
        
        model = StateSpaceModel(
            state_matrix=final_state_matrix, 
            output_matrix=output_matrix, 
            kalman_gain=kalman_gain
        )
        return SubspaceIdentificationResult(
            state_matrix=final_state_matrix, 
            output_matrix=output_matrix, 
            singular_values=singular_values, 
            models=[model]
        )
    
    else:
        # Multiple model case
        models = []
        for order in system_orders:
            # Compute state sequence
            state_sequence = (right_vectors_T[:order, :] @ inverse_sqrt_past_cov) @ past_data
            output_sequence = data[past_horizon:n_samples-future_horizon+1, :].T
            
            # Estimate matrices
            output_matrix = output_sequence @ np.linalg.pinv(state_sequence)
            
            n_time_steps = state_sequence.shape[1]
            augmented_regression_matrix = np.vstack([
                state_sequence[:, :n_time_steps-1], 
                data[past_horizon:n_samples-future_horizon, :].T
            ])
            next_states = state_sequence[:, 1:n_time_steps]
            
            system_kalman_matrix = next_states @ np.linalg.pinv(augmented_regression_matrix)
            state_matrix = system_kalman_matrix[:, :order]
            kalman_gain = system_kalman_matrix[:, order:order+n_channels]
            final_state_matrix = state_matrix + kalman_gain @ output_matrix
            
            models.append(StateSpaceModel(
                state_matrix=final_state_matrix, 
                output_matrix=output_matrix, 
                kalman_gain=kalman_gain
            ))
            
        return SubspaceIdentificationResult(
            state_matrix=None, 
            output_matrix=None, 
            singular_values=singular_values, 
            models=models
        )


def ssi1ca(
    signal_data: npt.NDArray[np.floating],
    params: Tuple[int, int, Union[int, List[int]]] 
) -> Dict[str, Any]:
    """
    Covariance-driven stochastic subspace identification.
    
    Python port of MATLAB ssi1ca.m that performs basic covariance-driven SSI
    using extended observability matrix with no particular SVD weighting.
    
    Args:
        signal_data: Multichannel data where columns are channels, shape (N, m)
        params: [future, past, order(s)] parameters
        
    Returns:
        Dictionary containing A, C matrices and singular values
    """
    N, m = signal_data.shape
    future, past = params[0], params[1]
    
    # Handle order specification
    if len(params) == 3:
        if isinstance(params[2], (list, np.ndarray)):
            orders = params[2]
        else:
            orders = [params[2]]
    else:
        orders = list(params[2:])
    
    if len(orders) == 0:
        raise ValueError("System order list is empty")
    
    if np.any(np.array(orders) > m * future):
        raise ValueError("Order request exceeds m*f")
    
    # Build data matrix
    n_cols = N - past - future + 1
    n_rows = m * (past + future)
    D = np.zeros((n_rows, n_cols))
    
    for kk in range(n_cols):
        block = signal_data[kk:kk+past+future, :].T.ravel()
        D[:, kk] = block
    
    # Split into past and future
    Yp = D[:m*past, :]
    Yf = D[m*past:, :]
    
    # Compute past-to-future projection
    Rfp = (Yf @ Yp.T) / n_cols
    U, S, Vt = np.linalg.svd(Rfp, full_matrices=False)
    sigma = np.diag(S)
    
    if len(orders) == 1:
        # Single model with given order
        n = orders[0]
        O = U[:, :n] @ np.diag(np.sqrt(sigma[:n]))
        
        # Split observability matrix to get C and A
        C = O[:m, :]
        if O.shape[0] > m:
            O1 = O[:-m, :]
            O2 = O[m:, :]
            # Solve for A: O2 = A * O1
            A = np.linalg.lstsq(O1.T, O2.T, rcond=None)[0].T
        else:
            A = np.zeros((n, n))
        
        return {
            'A': A,
            'C': C,
            'sigm': sigma,
            'order': n
        }
    else:
        # Multiple orders - return observability matrices for each
        results = {}
        for i, n in enumerate(orders):
            O = U[:, :n] @ np.diag(np.sqrt(sigma[:n]))
            C = O[:m, :]
            
            if O.shape[0] > m:
                O1 = O[:-m, :]
                O2 = O[m:, :]
                A = np.linalg.lstsq(O1.T, O2.T, rcond=None)[0].T
            else:
                A = np.zeros((n, n))
            
            results[f'order_{n}'] = {
                'A': A,
                'C': C,
                'sigm': sigma,
                'order': n
            }
        
        results['sigm'] = sigma
        return results


def ssicca(
    signal_data: npt.NDArray[np.floating],
    params: Tuple[int, int, Union[int, List[int]]],
    compute_kalman: bool = True
) -> Dict[str, Any]:
    """
    Stochastic subspace identification using canonical correlation analysis.
    
    Python port of MATLAB ssicca.m that performs SSI using CCA weighting
    for improved numerical properties and noise handling.
    
    Args:
        signal_data: Multichannel data where columns are channels, shape (N, m)
        params: [future, past, order(s)] parameters
        compute_kalman: Whether to compute Kalman gain matrix
        
    Returns:
        Dictionary containing A, C, K matrices and singular values
    """
    N, m = signal_data.shape
    future, past = params[0], params[1]
    
    # Handle order specification  
    if len(params) == 3:
        if isinstance(params[2], (list, np.ndarray)):
            orders = params[2]
        else:
            orders = [params[2]]
    else:
        orders = list(params[2:])
    
    if len(orders) == 0:
        raise ValueError("System order list is empty")
    
    if np.any(np.array(orders) > m * future):
        raise ValueError("Order request exceeds m*f")
    
    # Build data matrix
    n_cols = N - past - future + 1
    n_rows = m * (past + future)
    D = np.zeros((n_rows, n_cols))
    
    for kk in range(n_cols):
        block = signal_data[kk:kk+past+future, :].T.ravel()
        D[:, kk] = block
    
    # Split into past and future
    Yp = D[:m*past, :]
    Yf = D[m*past:, :]
    
    # Compute covariance matrices
    Rff = (Yf @ Yf.T) / n_cols
    Rpp = (Yp @ Yp.T) / n_cols
    Rfp = (Yf @ Yp.T) / n_cols
    
    # CCA weighting: compute inverse square roots
    try:
        # Eigendecomposition for matrix square root inverse
        U1, S1, Vt1 = np.linalg.svd(Rff)
        inv_sqrt_Rff = Vt1.T @ np.diag(1.0 / np.sqrt(S1)) @ Vt1
        
        U2, S2, Vt2 = np.linalg.svd(Rpp) 
        inv_sqrt_Rpp = Vt2.T @ np.diag(1.0 / np.sqrt(S2)) @ Vt2
        
        # CCA matrix
        M = inv_sqrt_Rff @ Rfp @ inv_sqrt_Rpp
        
    except np.linalg.LinAlgError:
        # Fallback to regularized version if matrices are singular
        reg_eps = 1e-10
        Rff_reg = Rff + reg_eps * np.eye(Rff.shape[0])
        Rpp_reg = Rpp + reg_eps * np.eye(Rpp.shape[0])
        
        U1, S1, Vt1 = np.linalg.svd(Rff_reg)
        inv_sqrt_Rff = Vt1.T @ np.diag(1.0 / np.sqrt(S1)) @ Vt1
        
        U2, S2, Vt2 = np.linalg.svd(Rpp_reg)
        inv_sqrt_Rpp = Vt2.T @ np.diag(1.0 / np.sqrt(S2)) @ Vt2
        
        M = inv_sqrt_Rff @ Rfp @ inv_sqrt_Rpp
    
    # SVD of CCA matrix
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    sigma = np.diag(S)
    
    if len(orders) == 1:
        # Single model
        n = orders[0]
        
        # Construct extended observability matrix with CCA weighting
        O = inv_sqrt_Rff @ U[:, :n] @ np.diag(np.sqrt(sigma[:n]))
        
        # Extract C and A matrices
        C = O[:m, :]
        if O.shape[0] > m:
            O1 = O[:-m, :]
            O2 = O[m:, :]
            A = np.linalg.lstsq(O1.T, O2.T, rcond=None)[0].T
        else:
            A = np.zeros((n, n))
        
        result = {
            'A': A,
            'C': C, 
            'sigm': sigma,
            'order': n
        }
        
        # Compute Kalman gain if requested
        if compute_kalman and O.shape[0] > m:
            try:
                residual = O2 - A @ O1
                Ree = (residual @ residual.T) / residual.shape[1]
                K = np.linalg.lstsq(C.T, np.eye(m), rcond=None)[0].T
                result['K'] = K
                result['Ree'] = Ree
            except np.linalg.LinAlgError:
                pass
        
        return result
    
    else:
        # Multiple orders
        results = {}
        for i, n in enumerate(orders):
            O = inv_sqrt_Rff @ U[:, :n] @ np.diag(np.sqrt(sigma[:n]))
            C = O[:m, :]
            
            if O.shape[0] > m:
                O1 = O[:-m, :]
                O2 = O[m:, :]
                A = np.linalg.lstsq(O1.T, O2.T, rcond=None)[0].T
            else:
                A = np.zeros((n, n))
            
            order_result = {
                'A': A,
                'C': C,
                'sigm': sigma,
                'order': n
            }
            
            if compute_kalman and O.shape[0] > m:
                try:
                    residual = O2 - A @ O1
                    Ree = (residual @ residual.T) / residual.shape[1]
                    K = np.linalg.lstsq(C.T, np.eye(m), rcond=None)[0].T
                    order_result['K'] = K
                    order_result['Ree'] = Ree
                except np.linalg.LinAlgError:
                    pass
            
            results[f'order_{n}'] = order_result
        
        results['sigm'] = sigma
        return results 
