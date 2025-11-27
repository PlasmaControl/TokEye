"""
TokEye: Automatic classification and localization of fluctuating signals.

TokEye is an open-source Python application for automatic classification and
localization of fluctuating signals in plasma physics using deep learning.
"""

from TokEye.exceptions import (
    CacheError,
    ConfigurationError,
    InvalidMaskError,
    InvalidModelError,
    InvalidSignalError,
    InvalidSpectrogramError,
    ModelError,
    ModelInferenceError,
    ModelLoadError,
    PostProcessError,
    ProcessingError,
    TilingError,
    TokEyeError,
    TransformError,
    ValidationError,
)

__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # Base exception
    "TokEyeError",
    # Configuration
    "ConfigurationError",
    # Validation
    "ValidationError",
    "InvalidSignalError",
    "InvalidSpectrogramError",
    "InvalidMaskError",
    # Processing
    "ProcessingError",
    "TransformError",
    "TilingError",
    "PostProcessError",
    # Model
    "ModelError",
    "ModelLoadError",
    "ModelInferenceError",
    "InvalidModelError",
    # Cache
    "CacheError",
]
