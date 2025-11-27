"""
Tests for UNet tiling and stitching utilities.

This module tests all functions in TokEye.processing.tiling, including:
- Spectrogram tiling
- Prediction stitching
- Roundtrip validation
"""

import numpy as np
import pytest

from TokEye.exceptions import InvalidSpectrogramError, TilingError
from TokEye.processing.tiling import (
    stitch_predictions,
    tile_spectrogram,
    validate_tiling_roundtrip,
)


class TestTileSpectrogram:
    """Test suite for tile_spectrogram function."""

    def test_basic_tiling_2d(self, sample_spectrogram_256):
        """Test basic tiling of 2D spectrogram."""
        tiles, metadata = tile_spectrogram(sample_spectrogram_256, tile_size=256)

        assert len(tiles) == 1  # Single 256x256 tile
        assert tiles[0].shape == (256, 256)
        assert metadata["original_width"] == 256
        assert metadata["original_height"] == 256
        assert metadata["num_tiles"] == 1
        assert metadata["padding"] == 0
        assert metadata["has_channels"] is False

    def test_tiling_large_spectrogram(self, sample_spectrogram_large):
        """Test tiling of large spectrogram (256x1024)."""
        tiles, metadata = tile_spectrogram(sample_spectrogram_large, tile_size=256)

        assert len(tiles) == 4  # 1024 / 256 = 4 tiles
        assert all(tile.shape == (256, 256) for tile in tiles)
        assert metadata["original_width"] == 1024
        assert metadata["num_tiles"] == 4
        assert metadata["padding"] == 0

    def test_tiling_with_padding(self):
        """Test tiling when width requires padding."""
        # 256x300 will need padding to fit into tiles
        spec = np.random.rand(256, 300).astype(np.float32)
        tiles, metadata = tile_spectrogram(spec, tile_size=256)

        assert len(tiles) == 2  # Need 2 tiles to cover 300 pixels
        assert all(tile.shape == (256, 256) for tile in tiles)
        assert metadata["padding"] > 0
        assert metadata["original_width"] == 300

    def test_tiling_with_overlap(self, sample_spectrogram_large):
        """Test tiling with overlap between adjacent tiles."""
        tiles_no_overlap, meta_no = tile_spectrogram(
            sample_spectrogram_large, tile_size=256, overlap=0
        )
        tiles_with_overlap, meta_with = tile_spectrogram(
            sample_spectrogram_large, tile_size=256, overlap=64
        )

        # With overlap, need more tiles to cover same area
        assert len(tiles_with_overlap) > len(tiles_no_overlap)
        assert meta_with["overlap"] == 64
        assert meta_with["stride"] == 192  # 256 - 64

    def test_tiling_3d_with_channels(self):
        """Test tiling of 3D spectrogram with channel dimension."""
        spec_3d = np.random.rand(3, 256, 512).astype(np.float32)
        tiles, metadata = tile_spectrogram(spec_3d, tile_size=256)

        assert len(tiles) == 2
        assert all(tile.shape == (3, 256, 256) for tile in tiles)
        assert metadata["has_channels"] is True
        assert metadata["num_channels"] == 3

    def test_metadata_completeness(self, sample_spectrogram_256):
        """Test that metadata contains all required keys."""
        tiles, metadata = tile_spectrogram(sample_spectrogram_256, tile_size=256)

        required_keys = [
            "original_width",
            "original_height",
            "tile_size",
            "overlap",
            "num_tiles",
            "padding",
            "stride",
            "has_channels",
            "num_channels",
        ]
        for key in required_keys:
            assert key in metadata

    def test_invalid_dimensionality(self):
        """Test that non-2D/3D spectrograms raise ValueError."""
        spec_1d = np.random.rand(256)
        with pytest.raises(InvalidSpectrogramError, match="must be 2D .* or 3D"):
            tile_spectrogram(spec_1d, tile_size=256)

        spec_4d = np.random.rand(2, 3, 256, 256)
        with pytest.raises(InvalidSpectrogramError, match="must be 2D .* or 3D"):
            tile_spectrogram(spec_4d, tile_size=256)

    def test_height_mismatch(self):
        """Test that height not matching tile_size raises ValueError."""
        spec = np.random.rand(128, 256)  # Height is 128, not 256
        with pytest.raises(TilingError, match="height .* must match tile_size"):
            tile_spectrogram(spec, tile_size=256)

    def test_negative_overlap(self, sample_spectrogram_256):
        """Test that negative overlap raises ValueError."""
        with pytest.raises(TilingError, match="Overlap must be non-negative"):
            tile_spectrogram(sample_spectrogram_256, tile_size=256, overlap=-10)

    def test_overlap_greater_than_tile_size(self, sample_spectrogram_256):
        """Test that overlap >= tile_size raises ValueError."""
        with pytest.raises(TilingError, match="Overlap .* must be less than tile_size"):
            tile_spectrogram(sample_spectrogram_256, tile_size=256, overlap=256)

        with pytest.raises(TilingError, match="Overlap .* must be less than tile_size"):
            tile_spectrogram(sample_spectrogram_256, tile_size=256, overlap=300)

    def test_zero_width(self):
        """Test that zero-width spectrogram raises ValueError."""
        spec = np.random.rand(256, 0)
        with pytest.raises(TilingError, match="width must be positive"):
            tile_spectrogram(spec, tile_size=256)

    def test_tile_independence(self, sample_spectrogram_large):
        """Test that modifying one tile doesn't affect others."""
        tiles, metadata = tile_spectrogram(sample_spectrogram_large, tile_size=256)

        # Modify first tile
        original_value = tiles[1][0, 0]
        tiles[0][0, 0] = 999.0

        # Second tile should be unchanged
        assert tiles[1][0, 0] == original_value

    def test_small_spectrogram(self):
        """Test tiling of spectrogram smaller than tile_size."""
        spec = np.random.rand(256, 100).astype(np.float32)
        tiles, metadata = tile_spectrogram(spec, tile_size=256)

        assert len(tiles) == 1
        assert tiles[0].shape == (256, 256)
        assert metadata["padding"] == 156  # 256 - 100


class TestStitchPredictions:
    """Test suite for stitch_predictions function."""

    def test_basic_stitching_2d(self, sample_spectrogram_256):
        """Test basic stitching of 2D predictions."""
        tiles, metadata = tile_spectrogram(sample_spectrogram_256, tile_size=256)

        # Simulate predictions (just use tiles as predictions)
        reconstructed = stitch_predictions(tiles, metadata)

        assert reconstructed.shape == sample_spectrogram_256.shape
        assert np.allclose(reconstructed, sample_spectrogram_256)

    def test_stitching_large_spectrogram(self, sample_spectrogram_large):
        """Test stitching of large spectrogram predictions."""
        tiles, metadata = tile_spectrogram(sample_spectrogram_large, tile_size=256)

        # Stitch back together
        reconstructed = stitch_predictions(tiles, metadata)

        assert reconstructed.shape == sample_spectrogram_large.shape
        assert np.allclose(reconstructed, sample_spectrogram_large)

    def test_stitching_with_overlap_blending(self, sample_spectrogram_large):
        """Test stitching with overlap blending enabled."""
        tiles, metadata = tile_spectrogram(
            sample_spectrogram_large, tile_size=256, overlap=64
        )

        # Stitch with blending
        reconstructed = stitch_predictions(tiles, metadata, blend_overlap=True)

        assert reconstructed.shape == sample_spectrogram_large.shape
        # With blending, overlapping regions are averaged
        assert np.allclose(reconstructed, sample_spectrogram_large, rtol=1e-5)

    def test_stitching_without_blending(self, sample_spectrogram_large):
        """Test stitching without overlap blending."""
        tiles, metadata = tile_spectrogram(
            sample_spectrogram_large, tile_size=256, overlap=64
        )

        # Stitch without blending (last tile wins)
        reconstructed = stitch_predictions(tiles, metadata, blend_overlap=False)

        assert reconstructed.shape == sample_spectrogram_large.shape

    def test_stitching_3d_with_channels(self):
        """Test stitching of 3D predictions with channels."""
        spec_3d = np.random.rand(3, 256, 512).astype(np.float32)
        tiles, metadata = tile_spectrogram(spec_3d, tile_size=256)

        reconstructed = stitch_predictions(tiles, metadata)

        assert reconstructed.shape == spec_3d.shape
        assert np.allclose(reconstructed, spec_3d)

    def test_padding_removal(self):
        """Test that padding is correctly removed during stitching."""
        spec = np.random.rand(256, 300).astype(np.float32)
        tiles, metadata = tile_spectrogram(spec, tile_size=256)

        reconstructed = stitch_predictions(tiles, metadata)

        # Should match original dimensions (padding removed)
        assert reconstructed.shape == spec.shape
        assert np.allclose(reconstructed, spec)

    def test_empty_tiles_list(self):
        """Test that empty tiles list raises ValueError."""
        metadata = {"num_tiles": 0}
        with pytest.raises(TilingError, match="Tiles list cannot be empty"):
            stitch_predictions([], metadata)

    def test_tile_count_mismatch(self, sample_spectrogram_256):
        """Test that tile count mismatch raises ValueError."""
        tiles, metadata = tile_spectrogram(sample_spectrogram_256, tile_size=256)

        # Modify metadata to expect more tiles
        metadata["num_tiles"] = 5

        with pytest.raises(
            TilingError, match="Number of tiles .* doesn't match metadata"
        ):
            stitch_predictions(tiles, metadata)

    def test_inconsistent_tile_shapes(self, sample_spectrogram_large):
        """Test that inconsistent tile shapes raise ValueError."""
        tiles, metadata = tile_spectrogram(sample_spectrogram_large, tile_size=256)

        # Make one tile the wrong shape
        tiles[1] = np.random.rand(128, 256)

        with pytest.raises(TilingError, match="Tile .* has shape .* expected"):
            stitch_predictions(tiles, metadata)

    def test_dtype_preservation(self, sample_spectrogram_256):
        """Test that dtype is preserved during stitching."""
        spec_float32 = sample_spectrogram_256.astype(np.float32)
        tiles, metadata = tile_spectrogram(spec_float32, tile_size=256)

        reconstructed = stitch_predictions(tiles, metadata)

        assert reconstructed.dtype == np.float32


class TestValidateTilingRoundtrip:
    """Test suite for validate_tiling_roundtrip function."""

    def test_roundtrip_no_overlap(self, sample_spectrogram_256):
        """Test roundtrip validation without overlap."""
        result = validate_tiling_roundtrip(
            sample_spectrogram_256, tile_size=256, overlap=0
        )
        assert result == True

    def test_roundtrip_with_overlap(self, sample_spectrogram_large):
        """Test roundtrip validation with overlap."""
        result = validate_tiling_roundtrip(
            sample_spectrogram_large, tile_size=256, overlap=64
        )
        assert result == True

    def test_roundtrip_3d(self):
        """Test roundtrip validation for 3D spectrograms."""
        spec_3d = np.random.rand(3, 256, 512).astype(np.float32)
        result = validate_tiling_roundtrip(spec_3d, tile_size=256, overlap=0)
        assert result == True

    def test_roundtrip_with_padding(self):
        """Test roundtrip validation when padding is needed."""
        spec = np.random.rand(256, 300).astype(np.float32)
        result = validate_tiling_roundtrip(spec, tile_size=256, overlap=0)
        assert result == True

    def test_roundtrip_custom_tolerance(self, sample_spectrogram_256):
        """Test roundtrip validation with custom tolerance."""
        result = validate_tiling_roundtrip(
            sample_spectrogram_256, tile_size=256, tolerance=1e-8
        )
        assert result == True


class TestIntegration:
    """Integration tests for tiling and stitching pipeline."""

    def test_multiple_sizes(self):
        """Test tiling and stitching with various spectrogram sizes."""
        widths = [256, 512, 768, 1024, 1500]

        for width in widths:
            spec = np.random.rand(256, width).astype(np.float32)
            tiles, metadata = tile_spectrogram(spec, tile_size=256)
            reconstructed = stitch_predictions(tiles, metadata)

            assert reconstructed.shape == spec.shape
            assert np.allclose(reconstructed, spec)

    def test_various_overlaps(self, sample_spectrogram_large):
        """Test tiling/stitching with different overlap values."""
        overlaps = [0, 32, 64, 128]

        for overlap in overlaps:
            tiles, metadata = tile_spectrogram(
                sample_spectrogram_large, tile_size=256, overlap=overlap
            )
            reconstructed = stitch_predictions(tiles, metadata, blend_overlap=True)

            assert reconstructed.shape == sample_spectrogram_large.shape
            assert np.allclose(reconstructed, sample_spectrogram_large, rtol=1e-5)

    def test_simulated_model_prediction(self, sample_spectrogram_large):
        """Test pipeline with simulated model predictions."""
        tiles, metadata = tile_spectrogram(sample_spectrogram_large, tile_size=256)

        # Simulate model predictions (apply simple threshold)
        predictions = [(tile > 0.5).astype(np.float32) for tile in tiles]

        # Stitch predictions
        full_prediction = stitch_predictions(predictions, metadata)

        assert full_prediction.shape == sample_spectrogram_large.shape
        assert np.all((full_prediction >= 0) & (full_prediction <= 1))

    def test_batch_processing(self):
        """Test processing multiple spectrograms in batch."""
        specs = [np.random.rand(256, 512).astype(np.float32) for _ in range(5)]

        for spec in specs:
            tiles, metadata = tile_spectrogram(spec, tile_size=256)
            reconstructed = stitch_predictions(tiles, metadata)
            assert np.allclose(reconstructed, spec)

    def test_edge_case_single_pixel_width(self):
        """Test edge case with minimal width spectrogram."""
        spec = np.random.rand(256, 1).astype(np.float32)
        tiles, metadata = tile_spectrogram(spec, tile_size=256)
        reconstructed = stitch_predictions(tiles, metadata)

        assert reconstructed.shape == spec.shape
        assert np.allclose(reconstructed, spec)

    def test_exact_multiples(self):
        """Test spectrograms with width exactly multiple of tile_size."""
        for num_tiles in [1, 2, 3, 4, 5]:
            width = 256 * num_tiles
            spec = np.random.rand(256, width).astype(np.float32)

            tiles, metadata = tile_spectrogram(spec, tile_size=256)
            assert len(tiles) == num_tiles
            assert metadata["padding"] == 0

            reconstructed = stitch_predictions(tiles, metadata)
            assert np.allclose(reconstructed, spec)
