"""
TokEye Processing Utilities Module

This module provides signal processing, model inference, and visualization utilities
for the TokEye plasma disruption detection system.
"""

from .transforms import apply_preemphasis, compute_stft, compute_wavelet
from .tiling import tile_spectrogram, stitch_predictions
from .inference import load_model, batch_inference
from .postprocess import apply_threshold, remove_small_objects, create_overlay
from .cache import generate_cache_key, CacheManager

__all__ = [
    # Transforms
    "apply_preemphasis",
    "compute_stft",
    "compute_wavelet",
    # Tiling
    "tile_spectrogram",
    "stitch_predictions",
    # Inference
    "load_model",
    "batch_inference",
    # Postprocessing
    "apply_threshold",
    "remove_small_objects",
    "create_overlay",
    # Caching
    "generate_cache_key",
    "CacheManager",
]
