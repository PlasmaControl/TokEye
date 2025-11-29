import logging

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sympy import Array

from TokEye.processing.postprocess import apply_threshold

logger = logging.getLogger(__name__)


def render_image(fig: plt.Figure) -> Image.Image:
    """Convert matplotlib figure to PIL Image."""
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img_array = buf.reshape(h, w, 4)
    img = Image.fromarray(img_array, mode="RGBA").convert("RGB")
    plt.close(fig)
    return img


def plot_array(arr: np.ndarray) -> plt.Figure:
    """Plot array as heatmap using consistent style."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(arr, aspect="auto", origin="lower", cmap="gist_heat")
    plt.axis("off")
    plt.tight_layout()
    return fig


def enhance(
    arr: np.ndarray,
    ch0_enabled: bool,
    ch1_enabled: bool,
    clip_min: float,
    clip_max: float,
) -> np.ndarray:
    """Create enhanced view with alpha transparency below clip_min.

    Args:
        arr_extract: 3D array (n_channels, height, width), values in [0, 1]
        ch0_enabled: Show channel 0 (green)
        ch1_enabled: Show channel 1 (red)
        clip_min: Values below this become transparent
        clip_max: Values above this are clamped

    Returns:
        RGB array ready for plotting
    """
    n_channels, h, w = arr.shape

    # Clip values
    arr[arr < clip_min / 100] = 0
    arr[arr > clip_max / 100] = 1

    # Create RGB image
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    # Set color channels based on enabled flags
    if ch0_enabled and n_channels > 0:
        rgb[:, :, 1] = arr[0]  # Green
    if ch1_enabled and n_channels > 1:
        rgb[:, :, 0] = arr[1]  # Red

    return rgb


def mask(
    arr: np.ndarray,
    ch0_enabled: bool,
    ch1_enabled: bool,
    threshold: float,
) -> np.ndarray:
    """Create binary mask view.

    Args:
        arr_extract: 3D array (n_channels, height, width), values in [0, 1]
        ch0_enabled: Show channel 0 (green)
        ch1_enabled: Show channel 1 (red)
        threshold: Binary threshold value

    Returns:
        RGB array ready for plotting
    """
    n_channels, h, w = arr.shape

    mask_ch0 = arr[0] > threshold
    mask_ch1 = arr[1] > threshold

    # Create RGB image
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    if ch0_enabled:
        rgb[:, :, 1] = mask_ch0  # Green
    if ch1_enabled:
        rgb[:, :, 0] = mask_ch1  # Red

    return rgb


def show_image(
    view_mode: str,
    arr: np.ndarray,
    arr_extract: np.ndarray | None,
    out_1_enabled: bool,
    out_2_enabled: bool,
    vmin: float,
    vmax: float,
    threshold: float,
) -> Image.Image | None:
    """Render visualization based on view mode.

    Args:
        view_mode: One of "Original", "Enhanced", or "Mask"
        arr: 2D array for original view
        arr_extract: 3D array (n_channels, height, width) for enhanced/mask views
        out_1_enabled: Enable channel 0 visualization
        out_2_enabled: Enable channel 1 visualization
        vmin: Min value for clipping (enhanced view)
        vmax: Max value for clipping (enhanced view)
        threshold: Threshold for binary mask (mask view)
    """
    try:
        if view_mode == "Original":
            display_arr = arr

        elif view_mode == "Enhanced":
            if arr_extract is None:
                logger.warning("No inference data for Enhanced view")
                return None
            display_arr = enhance(arr_extract, out_1_enabled, out_2_enabled, vmin, vmax)

        elif view_mode == "Mask":
            if arr_extract is None:
                logger.warning("No inference data for Mask view")
                return None
            display_arr = mask(arr_extract, out_1_enabled, out_2_enabled, threshold)

        else:
            logger.error(f"Unknown view mode: {view_mode}")
            return None

        fig = plot_array(display_arr)
        return render_image(fig)

    except Exception as e:
        logger.error(f"Visualization error: {e}")
        return None
