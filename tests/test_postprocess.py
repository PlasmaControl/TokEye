"""
Tests for post-processing and visualization utilities.

This module tests all functions in TokEye.processing.postprocess, including:
- Thresholding
- Small object removal
- Overlay creation
- Statistics computation
"""

import numpy as np
import pytest

from TokEye.exceptions import InvalidMaskError, PostProcessError
from TokEye.processing.postprocess import (
    apply_threshold,
    compute_channel_threshold_bounds,
    compute_statistics,
    create_overlay,
    remove_small_objects,
)


class TestApplyThreshold:
    """Test suite for apply_threshold function."""

    def test_basic_binary_threshold(self):
        """Test basic binary thresholding."""
        pred = np.array([[0.2, 0.8], [0.3, 0.9]])
        result = apply_threshold(pred, threshold=0.5, binary=True)

        expected = np.array([[0, 1], [0, 1]], dtype=np.uint8)
        assert np.array_equal(result, expected)

    def test_non_binary_threshold(self):
        """Test non-binary thresholding (keeps original values)."""
        pred = np.array([[0.2, 0.8], [0.3, 0.9]])
        result = apply_threshold(pred, threshold=0.5, binary=False)

        # Values below threshold become 0, above threshold stay same
        expected = np.array([[0.0, 0.8], [0.0, 0.9]])
        assert np.allclose(result, expected)

    def test_different_thresholds(self, sample_mask):
        """Test with various threshold values."""
        pred = sample_mask.astype(np.float32)

        # Low threshold - most pixels above
        result_low = apply_threshold(pred, threshold=0.1)
        # High threshold - most pixels below
        result_high = apply_threshold(pred, threshold=0.9)

        assert np.sum(result_low) >= np.sum(result_high)

    def test_threshold_warning(self):
        """Test that out-of-range threshold produces warning."""
        pred = np.random.rand(10, 10)

        with pytest.warns(RuntimeWarning, match="outside typical range"):
            apply_threshold(pred, threshold=1.5)

    def test_empty_prediction(self):
        """Test with empty prediction array."""
        empty = np.array([])
        result = apply_threshold(empty, threshold=0.5)
        assert result.shape == empty.shape

    def test_all_zeros(self):
        """Test with all-zero prediction."""
        pred = np.zeros((10, 10))
        result = apply_threshold(pred, threshold=0.5)
        assert np.all(result == 0)

    def test_all_ones(self):
        """Test with all-one prediction."""
        pred = np.ones((10, 10))
        result = apply_threshold(pred, threshold=0.5)
        assert np.all(result == 1)


class TestRemoveSmallObjects:
    """Test suite for remove_small_objects function."""

    def test_basic_removal(self):
        """Test basic small object removal."""
        # Create mask with one large and one small object
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:30, 10:30] = 1  # Large object (400 pixels)
        mask[80:85, 80:85] = 1  # Small object (25 pixels)

        cleaned, num_objects = remove_small_objects(mask, min_size=50)

        assert num_objects == 1  # Only large object remains
        assert np.sum(cleaned) == 400  # Large object area

    def test_no_removal_all_large(self):
        """Test when all objects are above min_size."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:40, 10:40] = 1  # 900 pixels
        mask[60:80, 60:80] = 1  # 400 pixels

        cleaned, num_objects = remove_small_objects(mask, min_size=100)

        assert num_objects == 2
        assert np.sum(cleaned) == np.sum(mask)

    def test_remove_all_small(self):
        """Test when all objects are below min_size."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:15, 10:15] = 1  # 25 pixels
        mask[80:85, 80:85] = 1  # 25 pixels

        cleaned, num_objects = remove_small_objects(mask, min_size=50)

        assert num_objects == 0
        assert np.sum(cleaned) == 0

    def test_connectivity_4_vs_8(self):
        """Test different connectivity modes."""
        # Diagonal connection
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2, 2] = 1
        mask[3, 3] = 1  # Diagonally connected

        # With 8-connectivity, these are one object
        cleaned_8, num_8 = remove_small_objects(mask, min_size=1, connectivity=8)
        assert num_8 == 1

        # With 4-connectivity, these are two objects
        cleaned_4, num_4 = remove_small_objects(mask, min_size=1, connectivity=4)
        assert num_4 == 2

    def test_invalid_connectivity(self):
        """Test that invalid connectivity raises ValueError."""
        mask = np.zeros((10, 10), dtype=np.uint8)

        with pytest.raises(InvalidMaskError, match="Connectivity must be 4 or 8"):
            remove_small_objects(mask, connectivity=6)

    def test_invalid_dimensionality(self):
        """Test that non-2D mask raises ValueError."""
        mask_1d = np.array([1, 0, 1])
        with pytest.raises(InvalidMaskError, match="Mask must be 2D"):
            remove_small_objects(mask_1d)

    def test_negative_min_size(self):
        """Test that negative min_size raises ValueError."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        with pytest.raises(InvalidMaskError, match="min_size must be non-negative"):
            remove_small_objects(mask, min_size=-5)

    def test_empty_mask(self):
        """Test with empty mask."""
        mask = np.array([[]]).reshape(0, 0).astype(np.uint8)
        cleaned, num_objects = remove_small_objects(mask)
        assert cleaned.shape == mask.shape
        assert num_objects == 0

    def test_dtype_conversion(self):
        """Test that float mask is converted correctly."""
        mask = np.array([[0.0, 1.0], [1.0, 0.0]])
        cleaned, num_objects = remove_small_objects(mask, min_size=1)
        assert cleaned.dtype == np.uint8


class TestCreateOverlay:
    """Test suite for create_overlay function."""

    def test_white_mode_basic(self):
        """Test white overlay mode."""
        spec = np.random.rand(50, 50)
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[20:30, 20:30] = 1

        overlay = create_overlay(spec, mask, mode="white", alpha=0.5)

        assert overlay.shape == (50, 50, 3)
        assert overlay.dtype == np.uint8
        # White regions should have high values in RGB
        assert np.all(overlay[25, 25] > 100)

    def test_hsv_mode(self):
        """Test HSV color overlay mode."""
        spec = np.random.rand(50, 50)
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:20, 10:20] = 1
        mask[30:40, 30:40] = 1

        overlay = create_overlay(spec, mask, mode="hsv", alpha=0.6)

        assert overlay.shape == (50, 50, 3)
        assert overlay.dtype == np.uint8

    def test_bicolor_mode(self):
        """Test bicolor overlay mode."""
        spec = np.random.rand(100, 100)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:40, 10:40] = 1  # Large object
        mask[80:85, 80:85] = 1  # Small object

        overlay = create_overlay(spec, mask, mode="bicolor", alpha=0.5)

        assert overlay.shape == (100, 100, 3)
        assert overlay.dtype == np.uint8

    def test_different_alpha_values(self):
        """Test different transparency levels."""
        spec = np.random.rand(30, 30)
        mask = np.ones((30, 30), dtype=np.uint8)

        overlay_opaque = create_overlay(spec, mask, alpha=1.0)
        overlay_transparent = create_overlay(spec, mask, alpha=0.0)
        overlay_half = create_overlay(spec, mask, alpha=0.5)

        # All should have same shape
        assert overlay_opaque.shape == overlay_transparent.shape == overlay_half.shape

    def test_shape_mismatch(self):
        """Test that shape mismatch raises ValueError."""
        spec = np.random.rand(50, 50)
        mask = np.zeros((60, 60), dtype=np.uint8)

        with pytest.raises(PostProcessError, match="doesn't match"):
            create_overlay(spec, mask)

    def test_invalid_alpha(self):
        """Test that invalid alpha raises ValueError."""
        spec = np.random.rand(10, 10)
        mask = np.zeros((10, 10), dtype=np.uint8)

        with pytest.raises(PostProcessError, match="Alpha must be"):
            create_overlay(spec, mask, alpha=-0.5)

        with pytest.raises(PostProcessError, match="Alpha must be"):
            create_overlay(spec, mask, alpha=1.5)

    def test_invalid_mode(self):
        """Test that invalid mode raises ValueError."""
        spec = np.random.rand(10, 10)
        mask = np.zeros((10, 10), dtype=np.uint8)

        with pytest.raises(PostProcessError, match="Invalid mode"):
            create_overlay(spec, mask, mode="invalid")

    def test_constant_spectrogram(self):
        """Test with constant spectrogram values."""
        spec = np.ones((30, 30)) * 0.5
        mask = np.zeros((30, 30), dtype=np.uint8)
        mask[10:20, 10:20] = 1

        overlay = create_overlay(spec, mask, mode="white")
        assert overlay.shape == (30, 30, 3)


class TestComputeChannelThresholdBounds:
    """Test suite for compute_channel_threshold_bounds function."""

    def test_basic_bounds(self):
        """Test basic threshold bounds computation."""
        pred = np.random.rand(50, 50)
        lower, upper = compute_channel_threshold_bounds(pred)

        assert 0 <= lower <= upper <= 1

    def test_uniform_prediction(self):
        """Test with uniform prediction values."""
        pred = np.ones((30, 30)) * 0.7
        lower, upper = compute_channel_threshold_bounds(pred)

        # For uniform prediction, bounds should be close
        assert abs(lower - upper) < 0.1

    def test_binary_prediction(self):
        """Test with binary prediction (0 and 1)."""
        pred = np.random.choice([0.0, 1.0], size=(50, 50))
        lower, upper = compute_channel_threshold_bounds(pred)

        assert 0 <= lower <= upper <= 1

    def test_invalid_dimensionality(self):
        """Test that non-2D prediction raises ValueError."""
        pred_1d = np.random.rand(100)
        with pytest.raises(InvalidMaskError, match="Prediction must be 2D"):
            compute_channel_threshold_bounds(pred_1d)

        pred_3d = np.random.rand(10, 10, 10)
        with pytest.raises(InvalidMaskError, match="Prediction must be 2D"):
            compute_channel_threshold_bounds(pred_3d)

    def test_empty_prediction(self):
        """Test with empty prediction."""
        pred = np.array([[]]).reshape(0, 0)
        lower, upper = compute_channel_threshold_bounds(pred)
        assert lower == 0.0 and upper == 1.0

    def test_different_num_steps(self):
        """Test with different numbers of threshold steps."""
        pred = np.random.rand(30, 30)

        lower_10, upper_10 = compute_channel_threshold_bounds(pred, num_steps=10)
        lower_100, upper_100 = compute_channel_threshold_bounds(pred, num_steps=100)

        # Results should be similar but not identical
        assert abs(lower_10 - lower_100) < 0.2
        assert abs(upper_10 - upper_100) < 0.2


class TestComputeStatistics:
    """Test suite for compute_statistics function."""

    def test_basic_statistics(self):
        """Test basic statistics computation."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:30, 10:30] = 1  # 400 pixels
        mask[60:75, 60:75] = 1  # 225 pixels

        stats = compute_statistics(mask, min_size=0)

        assert stats["num_objects"] == 2
        assert stats["total_area"] == 625
        assert stats["mean_area"] == 312.5
        assert stats["min_area"] == 225
        assert stats["max_area"] == 400

    def test_with_min_size_filter(self):
        """Test statistics with minimum size filtering."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:30, 10:30] = 1  # 400 pixels
        mask[80:85, 80:85] = 1  # 25 pixels

        # Without filter
        stats_all = compute_statistics(mask, min_size=0)
        assert stats_all["num_objects"] == 2

        # With filter (remove small object)
        stats_filtered = compute_statistics(mask, min_size=50)
        assert stats_filtered["num_objects"] == 1
        assert stats_filtered["total_area"] == 400

    def test_empty_mask(self):
        """Test with empty mask (no objects)."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        stats = compute_statistics(mask)

        assert stats["num_objects"] == 0
        assert stats["total_area"] == 0
        assert stats["mean_area"] == 0.0
        assert stats["coverage"] == 0.0

    def test_full_mask(self):
        """Test with full mask (all pixels set)."""
        mask = np.ones((50, 50), dtype=np.uint8)
        stats = compute_statistics(mask)

        assert stats["num_objects"] == 1
        assert stats["total_area"] == 2500
        assert stats["coverage"] == 1.0

    def test_coverage_calculation(self):
        """Test coverage fraction calculation."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[0:50, 0:50] = 1  # 25% coverage

        stats = compute_statistics(mask)
        assert abs(stats["coverage"] - 0.25) < 0.01

    def test_invalid_dimensionality(self):
        """Test that non-2D mask raises ValueError."""
        mask_1d = np.array([1, 0, 1])
        with pytest.raises(InvalidMaskError, match="Mask must be 2D"):
            compute_statistics(mask_1d)

    def test_median_calculation(self):
        """Test median area calculation."""
        mask = np.zeros((200, 200), dtype=np.uint8)
        # Create objects of sizes 100, 200, 300
        mask[0:10, 0:10] = 1  # 100 pixels
        mask[50:60, 50:70] = 1  # 200 pixels
        mask[100:110, 100:130] = 1  # 300 pixels

        stats = compute_statistics(mask)
        assert stats["median_area"] == 200.0


class TestIntegration:
    """Integration tests for post-processing pipeline."""

    def test_full_postprocess_pipeline(self):
        """Test complete post-processing pipeline."""
        # Simulate model prediction
        pred = np.random.rand(256, 256).astype(np.float32)

        # Apply threshold
        binary = apply_threshold(pred, threshold=0.5)

        # Remove small objects
        cleaned, num_objects = remove_small_objects(binary, min_size=50)

        # Compute statistics
        stats = compute_statistics(cleaned, min_size=0)

        # Create overlay
        spec = np.random.rand(256, 256)
        overlay = create_overlay(spec, cleaned, mode="hsv", alpha=0.6)

        assert overlay.shape == (256, 256, 3)
        assert stats["num_objects"] == num_objects

    def test_threshold_bounds_workflow(self):
        """Test workflow using threshold bounds."""
        pred = np.random.rand(100, 100)

        # Get bounds
        lower, upper = compute_channel_threshold_bounds(pred)

        # Apply threshold at midpoint
        mid_threshold = (lower + upper) / 2
        mask = apply_threshold(pred, threshold=mid_threshold)

        # Should have some objects
        cleaned, num_objects = remove_small_objects(mask, min_size=10)
        assert isinstance(num_objects, int)

    def test_multiple_overlay_modes(self):
        """Test creating overlays with all modes."""
        spec = np.random.rand(64, 64)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:30, 10:30] = 1
        mask[40:50, 40:50] = 1

        for mode in ["white", "bicolor", "hsv"]:
            overlay = create_overlay(spec, mask, mode=mode, alpha=0.5)
            assert overlay.shape == (64, 64, 3)
            assert overlay.dtype == np.uint8
