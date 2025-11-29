import logging

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from TokEye.processing.postprocess import apply_threshold

logger = logging.getLogger(__name__)


def render_original(spectrogram: np.ndarray) -> Image.Image:
    """Render original spectrogram."""
    if spectrogram is None:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    fig.canvas.draw()
    # Convert canvas to numpy array then to PIL Image
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img_array = buf.reshape(h, w, 4)
    img = Image.fromarray(img_array, mode="RGBA").convert("RGB")
    plt.close(fig)
    return img


def render_enhanced(
    inference: np.ndarray,
    ch0_enabled: bool,
    ch1_enabled: bool,
    clip_min: float,
    clip_max: float,
) -> Image.Image:
    """Render enhanced view with channel overlay."""
    if inference is None:
        return None

    # Clip values
    ch0 = np.clip(inference[0], clip_min, clip_max)
    ch1 = np.clip(inference[1], clip_min, clip_max)

    # Normalize to [0, 1]
    ch0 = (ch0 - clip_min) / (clip_max - clip_min) if clip_max > clip_min else ch0
    ch1 = (ch1 - clip_min) / (clip_max - clip_min) if clip_max > clip_min else ch1

    # Create RGB image
    rgb = np.zeros((*ch0.shape, 3))

    if ch0_enabled and ch1_enabled:
        rgb[:, :, 1] = ch0  # Green
        rgb[:, :, 0] = ch1  # Red
    elif ch0_enabled:
        rgb[:, :, 1] = ch0  # Green
    elif ch1_enabled:
        rgb[:, :, 0] = ch1  # Red

    rgb = (rgb * 255).astype(np.uint8)
    return Image.fromarray(rgb)


def render_mask(
    inference: np.ndarray,
    ch0_enabled: bool,
    ch1_enabled: bool,
    threshold: float,
) -> Image.Image:
    """Render binary mask view."""
    if inference is None:
        return None

    # Apply threshold
    mask_ch0 = apply_threshold(inference[0], threshold, binary=True)
    mask_ch1 = apply_threshold(inference[1], threshold, binary=True)

    # Create RGB image
    rgb = np.zeros((*mask_ch0.shape, 3))

    if ch0_enabled and ch1_enabled:
        rgb[:, :, 1] = mask_ch0  # Green
        rgb[:, :, 0] = mask_ch1  # Red
    elif ch0_enabled:
        rgb[:, :, 1] = mask_ch0  # Green
    elif ch1_enabled:
        rgb[:, :, 0] = mask_ch1  # Red

    rgb = (rgb * 255).astype(np.uint8)
    return Image.fromarray(rgb)


def show_image(
    view_mode: str,
    signal_transform: np.ndarray,
    inference_output: np.ndarray,
    out_1_enabled: bool,
    out_2_enabled: bool,
    vmin: float,
    vmax: float,
    threshold: float,
) -> Image.Image | None:
    """Render visualization based on view mode."""
    try:
        if view_mode == "Original":
            return render_original(signal_transform)
        elif view_mode == "Enhanced":
            return render_enhanced(
                inference_output,
                out_1_enabled,
                out_2_enabled,
                vmin,
                vmax,
            )
        elif view_mode == "Mask":
            return render_mask(
                inference_output,
                out_1_enabled,
                out_2_enabled,
                threshold,
            )
        return None
    except Exception as e:
        logger.error(f"Visualization error: {e}")
        return None
