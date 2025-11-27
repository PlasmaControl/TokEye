"""
UNet Tiling and Stitching Utilities

This module provides functions for splitting spectrograms into tiles for
UNet processing and reconstructing the full prediction from tiled outputs.
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

from TokEye.exceptions import InvalidSpectrogramError, TilingError


def tile_spectrogram(
    spectrogram: np.ndarray,
    tile_size: int,
    overlap: int = 0,
) -> Tuple[List[np.ndarray], Dict]:
    """
    Cut spectrogram into square tiles for UNet processing.

    The spectrogram is divided along the width (time axis) into height-sized
    squares. If the width is not evenly divisible, the last tile is padded
    to match tile_size.

    Args:
        spectrogram: Input spectrogram array of shape (height, width) or (channels, height, width)
        tile_size: Size of square tiles (must match spectrogram height)
        overlap: Number of pixels to overlap between adjacent tiles (default: 0)

    Returns:
        Tuple containing:
        - List of tile arrays, each of shape (height, tile_size) or (channels, height, tile_size)
        - Metadata dict with keys:
            - 'original_width': Original spectrogram width
            - 'original_height': Original spectrogram height
            - 'tile_size': Size of each tile
            - 'overlap': Overlap between tiles
            - 'num_tiles': Number of tiles created
            - 'padding': Amount of padding added to last tile
            - 'has_channels': Whether input has channel dimension
            - 'num_channels': Number of channels (if has_channels)

    Raises:
        ValueError: If spectrogram is not 2D or 3D
        ValueError: If tile_size doesn't match height
        ValueError: If overlap is negative or >= tile_size

    Example:
        >>> spec = np.random.randn(256, 1000)
        >>> tiles, metadata = tile_spectrogram(spec, tile_size=256)
        >>> print(len(tiles))  # 4
        >>> print(tiles[0].shape)  # (256, 256)
    """
    if spectrogram.ndim not in [2, 3]:
        raise InvalidSpectrogramError(
            f"Spectrogram must be 2D (H, W) or 3D (C, H, W), got {spectrogram.ndim}D"
        )

    if overlap < 0:
        raise TilingError(f"Overlap must be non-negative, got {overlap}")

    if overlap >= tile_size:
        raise TilingError(
            f"Overlap ({overlap}) must be less than tile_size ({tile_size})"
        )

    # Handle channel dimension
    has_channels = spectrogram.ndim == 3
    if has_channels:
        num_channels, height, width = spectrogram.shape
    else:
        height, width = spectrogram.shape
        num_channels = None

    if height != tile_size:
        raise TilingError(
            f"Spectrogram height ({height}) must match tile_size ({tile_size})"
        )

    if width == 0:
        raise TilingError("Spectrogram width must be positive")

    # Calculate stride (step size between tiles)
    stride = tile_size - overlap

    # Calculate number of tiles needed
    num_tiles = int(np.ceil((width - overlap) / stride))

    # Calculate total width needed (including padding)
    total_width_needed = (num_tiles - 1) * stride + tile_size
    padding = total_width_needed - width

    # Pad spectrogram if necessary
    if padding > 0:
        if has_channels:
            pad_width = ((0, 0), (0, 0), (0, padding))
        else:
            pad_width = ((0, 0), (0, padding))

        spectrogram_padded = np.pad(
            spectrogram, pad_width, mode="constant", constant_values=0
        )
    else:
        spectrogram_padded = spectrogram

    # Extract tiles
    tiles = []
    for i in range(num_tiles):
        start_idx = i * stride
        end_idx = start_idx + tile_size

        if has_channels:
            tile = spectrogram_padded[:, :, start_idx:end_idx]
        else:
            tile = spectrogram_padded[:, start_idx:end_idx]

        tiles.append(tile.copy())

    # Create metadata for stitching
    metadata = {
        "original_width": width,
        "original_height": height,
        "tile_size": tile_size,
        "overlap": overlap,
        "num_tiles": num_tiles,
        "padding": padding,
        "stride": stride,
        "has_channels": has_channels,
        "num_channels": num_channels,
    }

    return tiles, metadata


def stitch_predictions(
    tiles: List[np.ndarray],
    metadata: Dict,
    blend_overlap: bool = True,
) -> np.ndarray:
    """
    Reconstruct full prediction from tiles with optional overlap blending.

    Args:
        tiles: List of prediction tiles, each of shape matching tile_size from metadata
        metadata: Metadata dict returned from tile_spectrogram()
        blend_overlap: If True and overlap > 0, blend overlapping regions using averaging

    Returns:
        Reconstructed full prediction array with padding removed

    Raises:
        ValueError: If tiles list is empty
        ValueError: If number of tiles doesn't match metadata
        ValueError: If tile shapes are inconsistent

    Example:
        >>> spec = np.random.randn(256, 1000)
        >>> tiles, metadata = tile_spectrogram(spec, tile_size=256)
        >>> # Process tiles through model...
        >>> predictions = [model(tile) for tile in tiles]
        >>> full_prediction = stitch_predictions(predictions, metadata)
        >>> print(full_prediction.shape)  # (256, 1000)
    """
    if not tiles:
        raise TilingError("Tiles list cannot be empty")

    if len(tiles) != metadata["num_tiles"]:
        raise TilingError(
            f"Number of tiles ({len(tiles)}) doesn't match metadata ({metadata['num_tiles']})"
        )

    # Extract metadata
    tile_size = metadata["tile_size"]
    overlap = metadata["overlap"]
    stride = metadata["stride"]
    original_width = metadata["original_width"]
    original_height = metadata["original_height"]
    has_channels = metadata["has_channels"]
    num_channels = metadata["num_channels"]
    padding = metadata["padding"]

    # Validate tile shapes
    expected_shape = (
        (num_channels, tile_size, tile_size) if has_channels else (tile_size, tile_size)
    )

    for i, tile in enumerate(tiles):
        if tile.shape != expected_shape:
            raise TilingError(
                f"Tile {i} has shape {tile.shape}, expected {expected_shape}"
            )

    # Calculate total reconstructed width (including padding)
    reconstructed_width = (len(tiles) - 1) * stride + tile_size

    # Initialize output array
    if has_channels:
        output_shape = (num_channels, original_height, reconstructed_width)
    else:
        output_shape = (original_height, reconstructed_width)

    output = np.zeros(output_shape, dtype=tiles[0].dtype)

    # Handle overlap blending
    if overlap > 0 and blend_overlap:
        # Count how many tiles contribute to each pixel (for averaging)
        weight_map = np.zeros(output_shape, dtype=np.float32)

        # Stitch tiles with averaging in overlapping regions
        for i, tile in enumerate(tiles):
            start_idx = i * stride
            end_idx = start_idx + tile_size

            if has_channels:
                output[:, :, start_idx:end_idx] += tile
                weight_map[:, :, start_idx:end_idx] += 1.0
            else:
                output[:, start_idx:end_idx] += tile
                weight_map[:, start_idx:end_idx] += 1.0

        # Normalize by weight map (average overlapping regions)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            output = np.divide(output, weight_map, where=weight_map > 0)

    else:
        # Simple stitching without blending (last tile wins in overlap)
        for i, tile in enumerate(tiles):
            start_idx = i * stride
            end_idx = start_idx + tile_size

            if has_channels:
                output[:, :, start_idx:end_idx] = tile
            else:
                output[:, start_idx:end_idx] = tile

    # Remove padding
    if padding > 0:
        if has_channels:
            output = output[:, :, :-padding]
        else:
            output = output[:, :-padding]

    # Verify final shape matches original
    expected_final_shape = (
        (num_channels, original_height, original_width)
        if has_channels
        else (original_height, original_width)
    )

    if output.shape != expected_final_shape:
        warnings.warn(
            f"Stitched output shape {output.shape} doesn't match expected "
            f"{expected_final_shape}. This may indicate an issue with tiling.",
            RuntimeWarning,
        )

    return output


def validate_tiling_roundtrip(
    spectrogram: np.ndarray,
    tile_size: int,
    overlap: int = 0,
    tolerance: float = 1e-6,
) -> bool:
    """
    Validate that tiling and stitching correctly reconstruct the input.

    This is a utility function for testing the tiling/stitching pipeline.

    Args:
        spectrogram: Input spectrogram to test
        tile_size: Tile size for tiling
        overlap: Overlap between tiles
        tolerance: Maximum allowed absolute difference

    Returns:
        True if reconstruction matches input within tolerance

    Example:
        >>> spec = np.random.randn(256, 1000)
        >>> assert validate_tiling_roundtrip(spec, tile_size=256)
    """
    # Tile the spectrogram
    tiles, metadata = tile_spectrogram(spectrogram, tile_size, overlap)

    # Stitch back together
    reconstructed = stitch_predictions(tiles, metadata, blend_overlap=True)

    # Check if reconstruction matches original
    max_diff = np.max(np.abs(spectrogram - reconstructed))

    return max_diff < tolerance
