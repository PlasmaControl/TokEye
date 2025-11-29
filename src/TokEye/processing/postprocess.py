import cv2
import numpy as np


def apply_threshold(
    prediction: np.ndarray,
    threshold: float = 0.5,
    binary: bool = True,
) -> np.ndarray:
    """
    Apply threshold to prediction mask.

    Args:
        prediction: Input prediction array with values typically in [0, 1]
        threshold: Threshold value for binarization
        binary: If True, return binary mask (0 or 1), otherwise return thresholded values

    Returns:
        Thresholded prediction array

    Raises:
        ValueError: If threshold is not in valid range

    Example:
        >>> pred = np.random.rand(256, 256)
        >>> mask = apply_threshold(pred, threshold=0.5)
        >>> print(np.unique(mask))  # [0, 1]
    """
    if not 0 <= threshold <= 1:
        raise ValueError(f"Threshold must be in [0, 1], got {threshold}")

    if prediction.size == 0:
        return prediction.copy()

    if binary:
        # Binary thresholding
        thresholded = (prediction >= threshold).astype(np.uint8)
    else:
        # Keep original values above threshold, zero out below
        thresholded = np.where(prediction >= threshold, prediction, 0)

    return thresholded


def remove_small_objects(
    mask: np.ndarray,
    min_size: int = 50,
    connectivity: int = 8,
) -> tuple[np.ndarray, int]:
    """
    Remove small connected components from binary mask.

    Uses OpenCV's connected components analysis to identify and filter
    out small objects based on area.

    Args:
        mask: Binary input mask (values should be 0 or 1/255)
        min_size: Minimum object size in pixels to keep
        connectivity: Connectivity for connected components (4 or 8)

    Returns:
        Tuple containing:
        - Cleaned binary mask with small objects removed
        - Number of remaining objects

    Raises:
        ValueError: If mask is not 2D
        ValueError: If connectivity is not 4 or 8

    Example:
        >>> mask = np.random.randint(0, 2, (256, 256), dtype=np.uint8)
        >>> cleaned, num_objects = remove_small_objects(mask, min_size=100)
        >>> print(f"Found {num_objects} objects")
    """
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got {mask.ndim}D")

    if connectivity not in [4, 8]:
        raise ValueError(f"Connectivity must be 4 or 8, got {connectivity}")

    if min_size < 0:
        raise ValueError(f"min_size must be non-negative, got {min_size}")

    if mask.size == 0:
        return mask.copy(), 0

    # Ensure mask is binary uint8
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8)

    # Ensure mask is 0 or 255 for cv2
    mask = (mask > 0).astype(np.uint8) * 255

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=connectivity
    )

    # Stats columns: [left, top, width, height, area]
    # Label 0 is background, so start from 1
    areas = stats[1:, cv2.CC_STAT_AREA]

    # Create output mask
    cleaned_mask = np.zeros_like(mask, dtype=np.uint8)

    # Keep only objects larger than min_size
    num_kept = 0
    for i in range(1, num_labels):
        if areas[i - 1] >= min_size:
            cleaned_mask[labels == i] = 255
            num_kept += 1

    # Convert back to binary (0 or 1)
    cleaned_mask = (cleaned_mask > 0).astype(np.uint8)

    return cleaned_mask, num_kept


def create_overlay(
    spectrogram: np.ndarray,
    mask: np.ndarray,
    mode: str = "white",
    alpha: float = 0.5,
    coherent_color: tuple[int, int, int] = (0, 0, 255),  # Blue in BGR
    transient_color: tuple[int, int, int] = (0, 255, 0),  # Green in BGR
) -> np.ndarray:
    """
    Create visualization overlay of mask on spectrogram.

    Args:
        spectrogram: Input spectrogram array (2D)
        mask: Binary or labeled mask (2D)
        mode: Visualization mode:
            - 'white': Simple white overlay
            - 'bicolor': Blue for coherent, red for transient (requires labeled mask)
            - 'hsv': Unique colors per component using HSV color space
        alpha: Transparency of overlay (0=transparent, 1=opaque)
        coherent_color: BGR color for coherent structures (mode='bicolor')
        transient_color: BGR color for transient structures (mode='bicolor')

    Returns:
        RGB image with overlay (uint8 array of shape (H, W, 3))

    Raises:
        ValueError: If shapes don't match
        ValueError: If alpha is not in [0, 1]
        ValueError: If mode is invalid

    Example:
        >>> spec = np.random.randn(256, 256)
        >>> mask = np.random.randint(0, 2, (256, 256))
        >>> overlay = create_overlay(spec, mask, mode='white', alpha=0.5)
        >>> print(overlay.shape)  # (256, 256, 3)
    """
    if spectrogram.shape != mask.shape:
        raise ValueError(
            f"Spectrogram shape {spectrogram.shape} doesn't match mask shape {mask.shape}"
        )

    if not 0 <= alpha <= 1:
        raise ValueError(f"Alpha must be in [0, 1], got {alpha}")

    if mode not in ["white", "bicolor", "hsv"]:
        raise ValueError(f"Invalid mode '{mode}', must be 'white', 'bicolor', or 'hsv'")

    # Normalize spectrogram to [0, 255]
    spec_min, spec_max = spectrogram.min(), spectrogram.max()
    if spec_max - spec_min > 1e-10:
        spec_normalized = (
            (spectrogram - spec_min) / (spec_max - spec_min) * 255
        ).astype(np.uint8)
    else:
        spec_normalized = np.zeros_like(spectrogram, dtype=np.uint8)

    # Convert to BGR (grayscale to color)
    base_image = cv2.cvtColor(spec_normalized, cv2.COLOR_GRAY2BGR)

    # Create overlay based on mode
    if mode == "white":
        # Simple white overlay
        overlay = base_image.copy()
        overlay[mask > 0] = [255, 255, 255]  # White in BGR

    elif mode == "bicolor":
        # Bicolor overlay (requires some heuristic to distinguish coherent/transient)
        # For now, use a simple heuristic: larger components are coherent, smaller are transient
        # This would need to be customized based on actual classification

        try:
            num_labels, labels = cv2.connectedComponents(
                (mask > 0).astype(np.uint8), connectivity=8
            )
        except Exception as e:
            raise RuntimeError(f"Connected components analysis failed: {e}")

        overlay = base_image.copy()

        # Simple heuristic: classify by component size
        for i in range(1, num_labels):
            component_mask = labels == i
            component_size = np.sum(component_mask)

            # Threshold for classification (could be a parameter)
            if component_size > 100:  # Coherent (larger structures)
                overlay[component_mask] = coherent_color
            else:  # Transient (smaller structures)
                overlay[component_mask] = transient_color

    elif mode == "hsv":
        # HSV mode: assign unique color to each component
        try:
            num_labels, labels = cv2.connectedComponents(
                (mask > 0).astype(np.uint8), connectivity=8
            )
        except Exception as e:
            raise RuntimeError(f"Connected components analysis failed: {e}")

        # Create HSV image
        hsv_image = np.zeros((*spectrogram.shape, 3), dtype=np.uint8)
        hsv_image[:, :, 1] = 255  # Full saturation
        hsv_image[:, :, 2] = 255  # Full value

        # Assign hue based on component label
        for i in range(1, num_labels):
            component_mask = labels == i
            # Distribute hues evenly across spectrum
            hue = int((i - 1) * 179 / max(1, num_labels - 1))
            hsv_image[component_mask, 0] = hue

        # Convert HSV to BGR
        overlay_color = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

        # Blend with base image
        overlay = base_image.copy()
        mask_bool = mask > 0
        overlay[mask_bool] = overlay_color[mask_bool]

    # Blend overlay with base image using alpha
    result = cv2.addWeighted(base_image, 1 - alpha, overlay, alpha, 0)

    # Convert BGR to RGB for standard display
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    return result_rgb


def compute_channel_threshold_bounds(
    prediction: np.ndarray,
    num_steps: int = 100,
) -> tuple[float, float]:
    """
    Compute lower and upper threshold bounds for a single channel prediction.

    Lower bound: Highest threshold value where applying it results in no pixels (empty mask)
    Upper bound: Lowest threshold value where applying it covers the full predicted object

    Args:
        prediction: Single channel prediction array (2D) with values in [0, 1]
        num_steps: Number of threshold steps to test

    Returns:
        Tuple of (lower_bound, upper_bound) normalized to [0, 1]

    Example:
        >>> pred = np.random.rand(256, 256)
        >>> lower, upper = compute_channel_threshold_bounds(pred)
        >>> print(f"Threshold range: [{lower:.3f}, {upper:.3f}]")
    """
    if prediction.ndim != 2:
        raise ValueError(f"Prediction must be 2D, got {prediction.ndim}D")

    if prediction.size == 0:
        return 0.0, 1.0

    # Get the range of values in the prediction
    pred_min = float(prediction.min())
    pred_max = float(prediction.max())

    if pred_max - pred_min < 1e-10:
        # Uniform prediction, return middle value
        return pred_min, pred_max

    # Test threshold values from min to max
    thresholds = np.linspace(pred_min, pred_max, num_steps)

    # Find lower bound: highest threshold with no pixels
    lower_bound = pred_min
    for thresh in thresholds:
        mask = prediction >= thresh
        if np.any(mask):
            # Found first threshold that produces pixels
            break
        lower_bound = thresh

    # Find upper bound: lowest threshold that covers full object
    # We define "full object" as the pixels that would be detected at minimum threshold
    full_object_mask = (
        prediction >= pred_min + (pred_max - pred_min) * 0.01
    )  # 1% above minimum

    upper_bound = pred_max
    for thresh in reversed(thresholds):
        mask = prediction >= thresh
        # Check if this threshold covers the full object
        if np.array_equal(mask, full_object_mask) or np.all(mask[full_object_mask]):
            upper_bound = thresh
        else:
            break

    # Normalize to [0, 1] range
    if pred_max - pred_min > 1e-10:
        lower_norm = (lower_bound - pred_min) / (pred_max - pred_min)
        upper_norm = (upper_bound - pred_min) / (pred_max - pred_min)
    else:
        lower_norm = 0.0
        upper_norm = 1.0

    return lower_norm, upper_norm


def compute_statistics(
    mask: np.ndarray,
    min_size: int = 0,
) -> dict:
    """
    Compute statistics about detected objects in mask.

    Args:
        mask: Binary mask (2D array)
        min_size: Minimum object size to include in statistics

    Returns:
        Dictionary containing:
        - 'num_objects': Total number of objects
        - 'total_area': Total area of all objects (pixels)
        - 'mean_area': Mean object area
        - 'median_area': Median object area
        - 'min_area': Minimum object area
        - 'max_area': Maximum object area
        - 'coverage': Fraction of image covered by objects

    Example:
        >>> mask = np.random.randint(0, 2, (256, 256))
        >>> stats = compute_statistics(mask, min_size=50)
        >>> print(f"Found {stats['num_objects']} objects")
    """
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got {mask.ndim}D")

    # Ensure mask is binary uint8
    mask_binary = (mask > 0).astype(np.uint8) * 255

    # Find connected components
    num_labels, labels, stats_array, centroids = cv2.connectedComponentsWithStats(
        mask_binary, connectivity=8
    )

    # Extract areas (skip background label 0)
    areas = stats_array[1:, cv2.CC_STAT_AREA]

    # Filter by minimum size
    if min_size > 0:
        areas = areas[areas >= min_size]

    # Compute statistics
    if len(areas) > 0:
        statistics = {
            "num_objects": len(areas),
            "total_area": int(np.sum(areas)),
            "mean_area": float(np.mean(areas)),
            "median_area": float(np.median(areas)),
            "min_area": int(np.min(areas)),
            "max_area": int(np.max(areas)),
            "coverage": float(np.sum(areas) / (mask.shape[0] * mask.shape[1])),
        }
    else:
        statistics = {
            "num_objects": 0,
            "total_area": 0,
            "mean_area": 0.0,
            "median_area": 0.0,
            "min_area": 0,
            "max_area": 0,
            "coverage": 0.0,
        }

    return statistics
