"""
General utility functions for eigspec package.

This module provides common utility functions used throughout the eigspec package:
- Array and matrix manipulation utilities
- Mathematical helper functions
- Data validation and processing utilities
- Common computational routines

Based on utility functions scattered throughout the MATLAB eigspec toolbox:
- Various helper functions in eigspec_mmain.m
- Mathematical utilities in subspace identification functions
- Array processing utilities in modal analysis functions
- General computational helpers used across the MATLAB codebase
"""

from typing import Tuple
import numpy as np
from numpy.typing import ArrayLike, NDArray

def demean(x: NDArray[np.number], axis: int = 0, inplace: bool = False) -> NDArray[np.number]:
    """
    Remove mean from input data along specified axis.
    
    Args:
        x: Input data array
        axis: Axis along which to compute mean (default: 0)
        inplace: If True, modify array in place (default: False)
        
    Returns:
        Demeaned data array of same shape as input
        
    Example:
        >>> x = np.array([[1, 2], [3, 4]])
        >>> demean(x)
        array([[-1, -1],
               [ 1,  1]])
    """
    if not isinstance(x, np.ndarray):
        raise TypeError("Input must be a numpy array")
        
    if inplace:
        x -= np.mean(x, axis=axis, keepdims=True)
        return x
    return x - np.mean(x, axis=axis, keepdims=True)

def complex_to_real(z: NDArray[np.complexfloating]) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Convert complex array to real and imaginary parts.
    
    Args:
        z: Complex input array
        
    Returns:
        Tuple of (real_part, imag_part)
        
    Example:
        >>> z = np.array([1+2j, 3+4j])
        >>> real, imag = complex_to_real(z)
        >>> real
        array([1., 3.])
        >>> imag
        array([2., 4.])
    """
    if not np.iscomplexobj(z):
        raise TypeError("Input must be a complex array")
    return np.real(z), np.imag(z)

def real_to_complex(real_part: ArrayLike, imag_part: ArrayLike) -> NDArray[np.complexfloating]:
    """
    Convert real and imaginary parts to complex array.
    
    Args:
        real_part: Real part array
        imag_part: Imaginary part array
        
    Returns:
        Complex array
        
    Example:
        >>> real = np.array([1, 3])
        >>> imag = np.array([2, 4])
        >>> real_to_complex(real, imag)
        array([1.+2.j, 3.+4.j])
    """
    real_arr = np.asarray(real_part)
    imag_arr = np.asarray(imag_part)
    if real_arr.shape != imag_arr.shape:
        raise ValueError(f"Shape mismatch: real {real_arr.shape} != imag {imag_arr.shape}")
    return real_arr + 1j * imag_arr

def validate_dimensions(*arrays: NDArray) -> None:
    """
    Validate that input arrays have compatible dimensions.
    
    Args:
        *arrays: Variable number of numpy arrays to validate
        
    Raises:
        ValueError: If dimensions are incompatible
        TypeError: If any input is not a numpy array
        
    Example:
        >>> a = np.zeros((2, 3))
        >>> b = np.ones((2, 3))
        >>> validate_dimensions(a, b)  # No error
        >>> c = np.zeros((3, 2))
        >>> validate_dimensions(a, c)  # Raises ValueError
    """
    if not arrays:
        return
        
    if not all(isinstance(arr, np.ndarray) for arr in arrays):
        raise TypeError("All inputs must be numpy arrays")
        
    shape = arrays[0].shape
    mismatched = [(i, arr.shape) for i, arr in enumerate(arrays[1:], 1) 
                  if arr.shape != shape]
    
    if mismatched:
        details = [f"array[{i}]: {s}" for i, s in mismatched]
        raise ValueError(f"Array shapes must match. Found shape {shape} but got:\n" + 
                        "\n".join(details))

def is_positive_definite(matrix: NDArray[np.number], rtol: float = 1e-5) -> bool:
    """
    Check if a matrix is positive definite using Cholesky decomposition.
    
    Args:
        matrix: Square matrix to check
        rtol: Relative tolerance for numerical stability checks
        
    Returns:
        True if matrix is positive definite
        
    Example:
        >>> A = np.array([[2, -1], [-1, 2]])  # Positive definite
        >>> is_positive_definite(A)
        True
        >>> B = np.array([[1, 2], [2, 1]])  # Not positive definite
        >>> is_positive_definite(B)
        False
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Input must be a numpy array")
        
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {matrix.shape}")
        
    if not np.allclose(matrix, matrix.T, rtol=rtol):
        return False
        
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False 