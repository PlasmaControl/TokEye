"""
Custom exception classes for TokEye.

This module defines a hierarchy of custom exceptions used throughout the TokEye
package to provide clear, specific error messages for different failure modes.

Exception Hierarchy:
    TokEyeError (base)
    ├── ConfigurationError
    ├── ValidationError
    │   ├── InvalidSignalError
    │   ├── InvalidSpectrogramError
    │   └── InvalidMaskError
    ├── ProcessingError
    │   ├── TransformError
    │   ├── TilingError
    │   └── PostProcessError
    ├── ModelError
    │   ├── ModelLoadError
    │   ├── ModelInferenceError
    │   └── InvalidModelError
    └── CacheError
"""


class TokEyeError(Exception):
    """
    Base exception class for all TokEye errors.

    All custom exceptions in TokEye inherit from this class, allowing users
    to catch all TokEye-specific errors with a single except clause.
    """

    pass


# Configuration Errors
class ConfigurationError(TokEyeError):
    """
    Raised when there are issues with configuration files or settings.

    Examples:
        - Invalid YAML syntax in config files
        - Missing required configuration parameters
        - Conflicting configuration values
    """

    pass


# Validation Errors
class ValidationError(TokEyeError):
    """
    Base class for input validation errors.

    Raised when function inputs do not meet expected criteria (shape, type,
    value range, etc.).
    """

    pass


class InvalidSignalError(ValidationError):
    """
    Raised when signal data is invalid.

    Examples:
        - Wrong dimensionality (expected 1D or 2D)
        - Contains NaN or Inf values
        - Empty array
        - Wrong dtype
    """

    pass


class InvalidSpectrogramError(ValidationError):
    """
    Raised when spectrogram data is invalid.

    Examples:
        - Wrong shape (height doesn't match tile size)
        - Contains NaN or Inf values
        - Negative values in magnitude spectrogram
        - Empty array
    """

    pass


class InvalidMaskError(ValidationError):
    """
    Raised when mask data is invalid.

    Examples:
        - Non-binary values in binary mask
        - Shape mismatch with spectrogram
        - Wrong dtype
    """

    pass


# Processing Errors
class ProcessingError(TokEyeError):
    """
    Base class for signal processing errors.

    Raised when operations in the processing pipeline fail.
    """

    pass


class TransformError(ProcessingError):
    """
    Raised when transform operations fail.

    Examples:
        - STFT computation fails
        - Wavelet transform fails
        - Invalid transform parameters
        - Numerical instability
    """

    pass


class TilingError(ProcessingError):
    """
    Raised when spectrogram tiling or stitching fails.

    Examples:
        - Tile size larger than spectrogram
        - Invalid overlap parameter
        - Metadata mismatch during stitching
        - Shape incompatibility
    """

    pass


class PostProcessError(ProcessingError):
    """
    Raised when post-processing operations fail.

    Examples:
        - Invalid threshold value
        - Morphological operation fails
        - Overlay creation fails
    """

    pass


# Model Errors
class ModelError(TokEyeError):
    """
    Base class for neural network model errors.
    """

    pass


class ModelLoadError(ModelError):
    """
    Raised when model loading fails.

    Examples:
        - Model file not found
        - Corrupted model file
        - Incompatible model format
        - Missing required layers
    """

    pass


class ModelInferenceError(ModelError):
    """
    Raised when model inference fails.

    Examples:
        - GPU out of memory
        - Input shape mismatch
        - Runtime error during forward pass
    """

    pass


class InvalidModelError(ModelError):
    """
    Raised when model structure is invalid.

    Examples:
        - Model missing required methods
        - Incompatible architecture
        - Invalid number of channels
    """

    pass


# Cache Errors
class CacheError(TokEyeError):
    """
    Raised when cache operations fail.

    Examples:
        - Cache directory not writable
        - Corrupted cache file
        - Cache eviction fails
        - Invalid cache key
    """

    pass
