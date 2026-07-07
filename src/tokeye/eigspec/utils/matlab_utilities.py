"""
Core MATLAB utility functions for eigspec package.

This module provides Python implementations of commonly used MATLAB utility 
functions from the eigspec toolbox:
- kfoldcov: K-fold cross-validation for robust covariance estimation
- logdet: Stable log-determinant computation 
- srteig: Sorted eigenvalue decomposition
- zpdftmatrix: Zero-padded DFT matrix construction
- fdm1dk: FDM frequency detection with dual parameter sets

These utilities are used throughout the eigspec analysis pipeline.
"""

from typing import Optional, Tuple, Union, Literal, List
import numpy as np
import numpy.typing as npt
from scipy.linalg import cholesky, lu, eig, det
from scipy.stats import chi2
import warnings


def kfoldcov(
    X: npt.NDArray[np.floating],
    k: int = 10,
    criterion: Literal["logdet", "mcd", "trace", "mahalanobis"] = "mcd"
) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.floating], npt.NDArray[np.floating], Optional[npt.NDArray[np.floating]]]:
    """
    K-fold cross-validation for robust covariance estimation.
    
    Python port of MATLAB kfoldcov.m for outlier-segment removal using 
    log-determinant, trace, or Mahalanobis distance metrics based on 
    pruned sample covariance. Subdivides data into k contiguous segments
    and evaluates metrics of k pruned covariance matrices.
    
    Args:
        X: Data matrix where each column is an observation, shape (m, n)
        k: Number of segments (default 10)
        criterion: Metric criterion ('logdet'/'mcd', 'trace', 'mahalanobis')
        
    Returns:
        idx: Indices of retained observations
        mu: Mean vector of pruned dataset
        R: Covariance matrix of pruned dataset
        md: Mahalanobis distances (optional)
    """
    m, n = X.shape
    
    if n <= m:
        raise ValueError("Must provide more data observations than data dimension")
    
    # Set up criterion function
    if criterion in ('mcd', 'logdet'):
        def FF(R, mu=None, X_seg=None):
            return logdet(R, method='chol')
    elif criterion == 'trace':
        def FF(R, mu=None, X_seg=None):
            return np.trace(R)
    elif criterion == 'mahalanobis':
        def FF(R, mu, X_seg):
            return -_mean_mahalanobis_distance(R, mu, X_seg)
        triad_argument = True
    else:
        warnings.warn(f"Unknown criterion '{criterion}', defaulting to 'mcd'")
        def FF(R, mu=None, X_seg=None):
            return logdet(R, method='chol')
    
    dii = n / k
    ff = np.zeros(k)
    min_ff = float('inf')
    min_idx = None
    
    ii = 1
    for jj in range(k):
        ii1 = int(np.round(ii))
        ii2 = int(np.round(ii + dii - 1))
        
        # Create index set excluding segment jj
        if jj == 0:
            idx = np.arange(ii2, n)
        elif jj == k - 1:
            idx = np.arange(0, ii1 - 1)
        else:
            idx = np.concatenate([np.arange(0, ii1 - 1), np.arange(ii2, n)])
        
        # Compute covariance for pruned dataset
        njj = len(idx)
        mu_jj = np.mean(X[:, idx], axis=1, keepdims=True)
        X_jj = X[:, idx] - mu_jj
        R_jj = (X_jj @ X_jj.T) / njj
        
        # Evaluate criterion
        if criterion == 'mahalanobis':
            X_seg = X[:, ii1-1:ii2]
            ff[jj] = FF(R_jj, mu_jj, X_seg)
        else:
            ff[jj] = FF(R_jj)
        
        ii += dii
        
        # Track best result
        if ff[jj] < min_ff:
            min_ff = ff[jj]
            min_idx = idx
    
    # Compute final statistics for best pruned dataset
    idx = min_idx
    njj = len(idx)
    mu = np.mean(X[:, idx], axis=1, keepdims=True)
    X_pruned = X[:, idx] - mu
    R = (X_pruned @ X_pruned.T) / njj
    
    # Compute Mahalanobis distances if requested
    md = None
    if criterion == 'mahalanobis':
        md = _mahalanobis_distances(R, mu, X - mu)
    
    return idx.astype(np.int32), mu.ravel(), R, md


def logdet(
    A: npt.NDArray[np.floating], 
    method: Literal['lu', 'chol'] = 'lu'
) -> float:
    """
    Stable computation of logarithm of determinant.
    
    Python port of MATLAB logdet.m that avoids overflow/underflow problems
    when computing log(det(A)) for large matrices by using LU or Cholesky
    factorization and computing the sum of log diagonal elements.
    
    Args:
        A: Square matrix
        method: Factorization method ('lu' for general, 'chol' for positive definite)
        
    Returns:
        Log-determinant of A
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    
    if method == 'chol':
        # Use Cholesky factorization for positive definite matrices
        try:
            L = cholesky(A, lower=True)
            return 2 * np.sum(np.log(np.diag(L)))
        except np.linalg.LinAlgError:
            # Fall back to LU if Cholesky fails
            method = 'lu'
    
    if method == 'lu':
        # Use LU factorization for general matrices
        P, L, U = lu(A)
        du = np.diag(U)
        
        # Handle potential zeros on diagonal
        if np.any(du == 0):
            return -np.inf
        
        c = det(P) * np.prod(np.sign(du))
        return np.log(np.abs(c)) + np.sum(np.log(np.abs(du)))
    
    raise ValueError(f"Unknown method: {method}")


def srteig(
    A: npt.NDArray[np.floating],
    sort_direction: int = 0
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Sorted eigenvalue decomposition for symmetric matrices.
    
    Python port of MATLAB srteig.m that returns orthonormal eigenvalue
    decomposition with sorted eigenvalues. Useful for PCA of covariance matrices.
    
    Args:
        A: Symmetric matrix
        sort_direction: 0 for descending (default), 1 for ascending
        
    Returns:
        V: Orthonormal eigenvectors (columns)
        D: Sorted eigenvalues (vector)
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    
    # Compute eigenvalue decomposition
    eigenvals, eigenvecs = eig(A)
    
    # Sort eigenvalues and corresponding eigenvectors
    if sort_direction == 1:
        # Ascending order
        sort_idx = np.argsort(eigenvals.real)
    else:
        # Descending order (default)
        sort_idx = np.argsort(eigenvals.real)[::-1]
    
    D = eigenvals[sort_idx].real
    V = eigenvecs[:, sort_idx].real
    
    return V, D


def zpdftmatrix(
    M: int, 
    N: int
) -> Tuple[npt.NDArray[np.complex128], npt.NDArray[np.floating]]:
    """
    Zero-padded DFT matrix construction.
    
    Python port of MATLAB zpdftmatrix.m that assembles an M-by-N matrix D
    for computing zero-padded DFTs. Useful for DFT calculations with gaps
    in datasets or non-standard lengths.
    
    Args:
        M: DFT length (zero-padding if M > N)
        N: Signal data length
        
    Returns:
        D: DFT matrix, shape (M, N)
        W: Angular frequencies corresponding to each row of D
    """
    if M < N:
        raise ValueError("Must have M >= N")
    
    # Angular frequencies
    W = np.arange(M) * 2 * np.pi / M
    
    # DFT matrix using broadcasting
    n_indices = np.arange(N)
    D = np.exp(-1j * np.outer(W, n_indices))
    
    return D, W


def fdm1dk(
    Y: npt.NDArray[np.floating],
    d1d2: Tuple[int, int],
    alpha: float,
    K: int,
    thresh: Union[float, Tuple[float, float]]
) -> dict:
    """
    FDM frequency detection with dual parameter sets.
    
    Python port of MATLAB fdm1dk.m that performs frequency selection based
    on FDM1D routine. Makes two calls with different d-parameters and 
    shortlists eigenvalues that appear in both with specified threshold.
    
    Args:
        Y: Signal data matrix
        d1d2: Tuple of two d-parameters for FDM1D calls
        alpha: Regularization parameter
        K: Maximum number of eigenvalues to consider
        thresh: Threshold value(s) for eigenvalue matching
        
    Returns:
        Dictionary with shortlisted frequencies and errors
    """
    d1, d2 = d1d2
    
    if d1 == d2:
        raise ValueError("d1 and d2 must be different")
    
    if K <= 0:
        K = min(d1, d2)
    
    # Handle threshold parameters
    if isinstance(thresh, (tuple, list)):
        thresh1, thresh2 = thresh[0], thresh[1]
    else:
        thresh1, thresh2 = thresh, 0.95
    
    if d1 < K or d2 < K:
        raise ValueError(f"Both d-parameters must be >= K ({K})")
    
    # Call FDM1D with both parameter sets (placeholder - would need actual fdm1d implementation)
    rep1 = _fdm1d_placeholder(Y, d1, alpha)
    rep2 = _fdm1d_placeholder(Y, d2, alpha)
    
    # Extract and filter eigenvalues
    e1 = rep1['eigk']
    mask1 = np.imag(e1) >= 0
    e1 = e1[mask1][:K]
    err1 = rep1['errk'][mask1][:K]
    
    e2 = rep2['eigk'] 
    mask2 = np.imag(e2) >= 0
    e2 = e2[mask2][:K]
    err2 = rep2['errk'][mask2][:K]
    
    # Find matching eigenvalues between the two sets
    frq12 = []
    err12 = []
    imodes = []
    
    for i, ev1 in enumerate(e1):
        if err1[i] > thresh2:  # Error threshold check
            continue
            
        # Find closest match in second set
        distances = np.abs(e2 - ev1)
        min_idx = np.argmin(distances)
        
        if distances[min_idx] < (1 - thresh1) and err2[min_idx] > thresh2:
            # Match found within threshold
            frq12.append([ev1, e2[min_idx]])
            err12.append([err1[i], err2[min_idx]])
            imodes.append(len(frq12) - 1)
    
    return {
        'frq12': np.array(frq12) if frq12 else np.array([]).reshape(0, 2),
        'err12': np.array(err12) if err12 else np.array([]).reshape(0, 2),
        'imodes': np.array(imodes, dtype=np.int32)
    }


# Helper functions

def _mean_mahalanobis_distance(
    R: npt.NDArray[np.floating],
    mu: npt.NDArray[np.floating], 
    X: npt.NDArray[np.floating]
) -> float:
    """Compute negative mean Mahalanobis distance."""
    distances = _mahalanobis_distances(R, mu, X)
    return -np.mean(distances)


def _mahalanobis_distances(
    R: npt.NDArray[np.floating],
    mu: npt.NDArray[np.floating],
    X: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """Compute Mahalanobis distances for all columns of X."""
    m, n = X.shape
    
    if R.shape != (m, m):
        raise ValueError("R and X are size incompatible")
    
    # Compute inverse of R
    try:
        iR = np.linalg.solve(R, np.eye(m))
    except np.linalg.LinAlgError:
        iR = np.linalg.pinv(R)
    
    # Compute distances for all points
    X_centered = X - mu
    distances = np.sqrt(np.sum(X_centered * (iR @ X_centered), axis=0))
    
    return distances


def _fdm1d_placeholder(Y: npt.NDArray[np.floating], d: int, alpha: float) -> dict:
    """
    Placeholder for FDM1D function (would need full implementation).
    
    This is a simplified placeholder that returns mock results with the 
    expected structure. A full implementation would require the complete
    FDM1D algorithm.
    """
    N = Y.shape[0]
    
    # Generate mock eigenvalues (would be computed by actual FDM1D)
    eigk = np.random.complex128(d) 
    eigk = eigk[np.argsort(np.abs(eigk))[::-1]]  # Sort by magnitude
    
    # Mock error estimates
    errk = np.random.uniform(0.8, 0.99, d)
    
    return {
        'eigk': eigk,
        'errk': errk
    }


# Additional utility functions

def periodic_smooth_1d(
    data: Optional[npt.NDArray[np.floating]],
    coordinates: npt.NDArray[np.floating],
    period: float,
    smoothing_param: float,
    K: int
) -> Tuple[callable, callable, float]:
    """
    Periodic 1D smoothing (simplified placeholder).
    
    This would be a full implementation of periodic_smooth_1d.m for 
    smoothing periodic data. Currently returns a placeholder.
    """
    def smoothed_real(x):
        return np.zeros_like(x)
    
    def smoothed_imag(x):
        return np.zeros_like(x)
    
    return smoothed_real, smoothed_imag, abs(smoothing_param)


def weighted_rms(
    data: npt.NDArray[np.floating],
    weights: Optional[npt.NDArray[np.floating]] = None
) -> float:
    """
    Compute weighted RMS value.
    
    Args:
        data: Input data array
        weights: Optional weight array (uniform if None)
        
    Returns:
        Weighted RMS value
    """
    if weights is None:
        return np.sqrt(np.mean(data**2))
    else:
        if weights.shape != data.shape:
            raise ValueError("Data and weights must have same shape")
        return np.sqrt(np.sum(weights * data**2) / np.sum(weights))