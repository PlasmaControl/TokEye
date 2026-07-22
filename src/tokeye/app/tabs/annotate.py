"""
Mask Annotation Interface Tab for TokEye

This module provides an interface for annotating masks over backdrop images.
The backdrop (loaded from .npy files) is read-only and used as reference only.
Only the mask layer is editable and saved.
"""

from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image


def pil_to_numpy(img: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy array."""
    return np.array(img)


def to_display_uint8(arr: np.ndarray) -> np.ndarray:
    """Robust display normalization: floats stretched 1st-99th percentile
    to 0-255; uint8 passthrough.

    Naively casting float data to uint8 (e.g. `arr.astype(np.uint8)`) silently
    truncates anything above 255 and clips everything else to a razor-thin
    band near 0, which is why log-scaled spectrograms (values roughly 0-10)
    used to render as near-black images. Percentile stretching instead maps
    the bulk of the data across the full display range.
    """
    if arr.dtype == np.uint8:
        return arr

    arr = np.asarray(arr, dtype=np.float64)
    if not np.isfinite(arr).any():
        # All-NaN/inf input - nothing meaningful to display.
        return np.zeros(arr.shape, dtype=np.uint8)

    lo, hi = np.nanpercentile(arr, [1, 99])
    if not hi > lo:
        # Flat (or degenerate) image - nothing to stretch.
        return np.zeros(arr.shape, dtype=np.uint8)

    scaled = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    scaled = np.nan_to_num(scaled, nan=0.0)  # stray NaNs render as black
    return (scaled * 255).astype(np.uint8)


def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert numpy array to PIL Image with RGBA support."""
    # Normalize to 0-255 if needed
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        arr = to_display_uint8(arr)
    elif arr.dtype != np.uint8:
        # Clip to valid range
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    # Convert to RGBA for transparency support
    if arr.ndim == 2:
        # Grayscale -> RGBA
        rgb = np.stack([arr, arr, arr], axis=-1)
        alpha = np.full(arr.shape, 255, dtype=np.uint8)
        rgba = np.dstack([rgb, alpha])
    elif arr.ndim == 3:
        if arr.shape[2] == 3:
            # RGB -> RGBA
            alpha = np.full(arr.shape[:2], 255, dtype=np.uint8)
            rgba = np.dstack([arr, alpha])
        elif arr.shape[2] == 4:
            # Already RGBA
            rgba = arr
        else:
            raise ValueError(f"Unsupported number of channels: {arr.shape[2]}")
    else:
        raise ValueError(f"Unsupported array dimensions: {arr.ndim}")

    return Image.fromarray(rgba, mode="RGBA")


# ============================================================================
# Backdrop and Mask Loading Functions
# ============================================================================


def find_existing_mask(npy_filename: str) -> str | None:
    """
    Find if there's an existing mask annotation for this npy file.

    Args:
        npy_filename: Name of the .npy file (e.g., "signal_20240115.npy")

    Returns:
        Path to mask file if exists, None otherwise
    """
    annotations_dir = Path("annotations")
    if not annotations_dir.exists():
        return None

    # Look for mask with same base name
    base_name = Path(npy_filename).stem

    # Try different mask naming patterns
    patterns = [
        f"{base_name}_mask.npy",
        f"{base_name}_annotation.npy",
        f"annotation_{base_name}.npy",
    ]

    for pattern in patterns:
        mask_path = annotations_dir / pattern
        if mask_path.exists():
            return str(mask_path)

    return None


def load_npy_as_backdrop(
    npy_file,
) -> tuple[
    Image.Image | None,
    np.ndarray | None,
    np.ndarray | None,
    str,
    str | None,
]:
    """
    Load .npy file as backdrop image and look for existing mask.

    Returns:
        (backdrop_image, backdrop_array, mask_array, info_text, npy_filename)
    """
    if npy_file is None:
        return None, None, None, "No file uploaded", None

    try:
        # Get the filename
        npy_filename = Path(npy_file.name).name

        # Load the array
        arr = np.load(npy_file.name)

        # Validate
        if arr.ndim not in [2, 3]:
            return (
                None,
                None,
                None,
                f"Error: Array must be 2D or 3D for annotation, got {arr.ndim}D",
                None,
            )

        # Convert to backdrop image
        backdrop_img = numpy_to_pil(arr)

        # Look for existing mask
        existing_mask_path = find_existing_mask(npy_filename)

        if existing_mask_path:
            # Load existing mask
            mask_arr = np.load(existing_mask_path)

            # Validate mask shape matches backdrop
            expected_shape = arr.shape[:2] if arr.ndim == 3 else arr.shape
            if mask_arr.shape[:2] != expected_shape:
                info = f"""
**Warning: Mask shape mismatch**
- Backdrop: {npy_filename}
- Backdrop shape: {arr.shape}
- Mask shape: {mask_arr.shape}
- Creating new empty mask instead
"""
                mask_arr = np.zeros(expected_shape, dtype=np.uint8)
            else:
                info = f"""
**Loaded Successfully:**
- Backdrop: {npy_filename}
- Shape: {arr.shape}
- Data type: {arr.dtype}
- Existing mask found: {Path(existing_mask_path).name}
- Mask shape: {mask_arr.shape}
- Ready for editing
"""
        else:
            # Create empty mask (same spatial dimensions as backdrop)
            if arr.ndim == 2:
                mask_arr = np.zeros(arr.shape, dtype=np.uint8)
            else:  # 3D
                mask_arr = np.zeros(arr.shape[:2], dtype=np.uint8)

            info = f"""
**Loaded Successfully:**
- Backdrop: {npy_filename}
- Shape: {arr.shape}
- Data type: {arr.dtype}
- No existing mask found - created empty mask
- Mask shape: {mask_arr.shape}
- Ready for annotation
"""

        return backdrop_img, arr, mask_arr, info, npy_filename

    except Exception as e:
        return None, None, None, f"Error loading file: {str(e)}", None


def create_composite_image(
    backdrop_img: Image.Image, mask_arr: np.ndarray
) -> Image.Image:
    """
    Create composite image with backdrop and mask overlay.

    Args:
        backdrop_img: Backdrop image (RGBA)
        mask_arr: Mask array (2D, values 0-255)

    Returns:
        Composite image with red semi-transparent mask overlay
    """
    composite = backdrop_img.copy().convert("RGBA")

    # Create red overlay where mask is non-zero
    if mask_arr.max() > 0:
        width, height = composite.size

        # Vectorized red, ~50%-alpha overlay (replaces an O(H*W) putpixel
        # double loop that hung for minutes on realistic image sizes).
        overlay = np.zeros((height, width, 4), dtype=np.uint8)
        overlay[..., 0] = 255  # red

        # Pad/crop the mask to the backdrop size instead of resizing, so
        # out-of-bounds regions stay transparent (matches the loop this
        # replaces, which only touched pixels within both bounds).
        rows = min(mask_arr.shape[0], height)
        cols = min(mask_arr.shape[1], width)
        overlay[:rows, :cols, 3] = np.where(mask_arr[:rows, :cols] > 0, 128, 0)

        red_overlay = Image.fromarray(overlay, mode="RGBA")

        # Composite the overlay
        composite = Image.alpha_composite(composite, red_overlay)

    return composite


def save_mask_annotation(
    mask_arr: np.ndarray | None, npy_filename: str, format_choice: str = "npy"
) -> str | None:
    """
    Save ONLY the mask annotation (not the backdrop image).

    Args:
        mask_arr: Mask array to save
        npy_filename: Original .npy filename for naming the mask
        format_choice: 'npy' or 'png'

    Returns:
        Saved filepath or None
    """
    if mask_arr is None:
        gr.Warning("No mask to save")
        return None

    try:
        annotations_dir = Path("annotations")
        annotations_dir.mkdir(exist_ok=True)

        # Use original filename as base
        base_name = Path(npy_filename).stem

        if format_choice == "npy":
            filepath = annotations_dir / f"{base_name}_mask.npy"
            np.save(filepath, mask_arr)
        else:  # png
            filepath = annotations_dir / f"{base_name}_mask.png"
            # Save mask as grayscale image
            mask_img = Image.fromarray(mask_arr, mode="L")
            mask_img.save(filepath)

        gr.Info(f"Mask saved to {filepath}")
        return str(filepath)

    except Exception as e:
        gr.Warning(f"Failed to save mask: {str(e)}")
        return None


# ============================================================================
# Mask Extraction and Processing
# ============================================================================


def _mask_from_layers(layers: list) -> np.ndarray | None:
    """Union, across stroke layers, of pixels with any alpha - color-agnostic.

    `gr.ImageEditor` stroke layers are RGBA images that are fully transparent
    everywhere except where the user drew, regardless of brush color. Using
    alpha (instead of thresholding a specific channel) means red, green,
    blue, and white strokes are all captured the same way.
    """
    mask_bool = None
    for layer in layers:
        if isinstance(layer, Image.Image):
            layer_arr = pil_to_numpy(layer)
        else:
            layer_arr = np.array(layer)

        if layer_arr.ndim == 3 and layer_arr.shape[2] == 4:
            stroke = layer_arr[:, :, 3] > 0
        elif layer_arr.ndim == 3:
            stroke = layer_arr.max(axis=-1) > 0
        else:
            stroke = layer_arr > 0

        mask_bool = stroke if mask_bool is None else (mask_bool | stroke)

    if mask_bool is None:
        return None
    return (mask_bool.astype(np.uint8)) * 255


def _mask_from_composite_diff(
    composite, backdrop_arr: np.ndarray | None
) -> np.ndarray:
    """Fallback: diff the flattened composite against the (normalized)
    backdrop, thresholding the max absolute difference across RGB channels.

    Color-agnostic (unlike the old red-channel-only diff) and correct for
    float backdrops, since both sides are normalized with the same
    `to_display_uint8` before comparison.
    """
    if isinstance(composite, Image.Image):
        modified_arr = pil_to_numpy(composite)
    else:
        modified_arr = np.array(composite)

    if modified_arr.ndim != 3:
        return (modified_arr > 128).astype(np.uint8) * 255

    canvas_rgb = modified_arr[:, :, :3].astype(np.int16)

    if backdrop_arr is None:
        # No backdrop to diff against - just look for bright brush strokes.
        return (canvas_rgb.max(axis=-1) > 128).astype(np.uint8) * 255

    backdrop_disp = to_display_uint8(backdrop_arr)
    if backdrop_disp.ndim == 3:
        backdrop_disp = backdrop_disp[:, :, 0]

    if backdrop_disp.shape != canvas_rgb.shape[:2]:
        backdrop_img = Image.fromarray(backdrop_disp)
        backdrop_img = backdrop_img.resize(
            (canvas_rgb.shape[1], canvas_rgb.shape[0])
        )
        backdrop_disp = np.array(backdrop_img)

    backdrop_rgb = np.stack([backdrop_disp] * 3, axis=-1).astype(np.int16)
    diff = np.abs(canvas_rgb - backdrop_rgb).max(axis=-1)
    return (diff > 30).astype(np.uint8) * 255


def extract_mask_from_canvas(
    canvas_output, backdrop_arr: np.ndarray | None
) -> np.ndarray | None:
    """
    Extract only the mask layer from the canvas, removing the backdrop.

    Priority order:
    1. Layers-first: if `canvas_output` is a dict with non-empty `layers`,
       the mask is the union over layers of `alpha > 0` - color-agnostic, so
       red/green/blue/white brush strokes are all captured.
    2. Fallback: a dict without layers (or a plain composite array) diffs
       the composite against the normalized backdrop, thresholding the max
       absolute difference across RGB channels.

    Args:
        canvas_output: Output from gr.ImageEditor
        backdrop_arr: Original backdrop array for comparison

    Returns:
        Binary mask array (0 or 255)
    """
    if canvas_output is None:
        return None

    try:
        if isinstance(canvas_output, dict):
            layers = canvas_output.get("layers") or []
            if layers:
                mask = _mask_from_layers(layers)
                if mask is not None:
                    return mask

            composite = canvas_output.get("composite")
            if composite is None:
                composite = canvas_output.get("background")
            if composite is None:
                return None
        else:
            composite = canvas_output

        return _mask_from_composite_diff(composite, backdrop_arr)

    except Exception as e:
        print(f"Error extracting mask: {e}")
        return None


def handle_save_mask(
    canvas_output, backdrop_arr: np.ndarray | None, filename: str | None, save_fmt: str
):
    """Extract + save the mask, returning a status message and a value for
    the `gr.File` download slot.

    Returns:
        (status_text, filepath) on success, or (status_text, None) on
        failure/empty save so the download component CLEARS. (A bare
        `gr.update()` would be gradio's skip sentinel, leaving a previous
        successful save's file visible as a stale download link.)
    """
    if canvas_output is None or filename is None:
        return "Error: No annotation to save", None

    try:
        mask_arr = extract_mask_from_canvas(canvas_output, backdrop_arr)
        if mask_arr is None:
            return "Error: Could not extract mask from canvas", None

        filepath = save_mask_annotation(mask_arr, filename, save_fmt)
        if filepath:
            return f"Mask saved successfully to: {filepath}", filepath
        return "Save failed", None
    except Exception as e:
        return f"Error: {str(e)}", None


# ============================================================================
# Gradio Interface
# ============================================================================


def annotate_tab():
    """Create the annotation tab interface."""

    with gr.Column() as tab:
        gr.Markdown("# Mask Annotation Interface")
        gr.Markdown(
            "Load .npy files (e.g., spectrograms) and annotate masks. The image serves as a backdrop only."
        )

        # State variables
        backdrop_array_state = gr.State(None)  # Original .npy array (read-only)
        mask_array_state = gr.State(None)  # Editable mask
        npy_filename_state = gr.State(None)  # Track filename for saving

        with gr.Row():
            # Left column: Controls
            with gr.Column(scale=1):
                gr.Markdown("### Load Spectrogram")

                npy_file_input = gr.File(
                    label="Upload .npy File (spectrogram/transform output)",
                    file_types=[".npy"],
                )

                load_btn = gr.Button("Load for Annotation", variant="primary")

                file_info = gr.Markdown("*No file loaded*")

                gr.Markdown("### Annotation Instructions")
                gr.Markdown("""
1. Load a .npy file (spectrogram or transform output)
2. The system will check for existing mask annotations
3. Draw or erase to create/edit the mask
4. Save the mask (image backdrop is never modified)

**Note**: Only the mask layer is saved, not the backdrop image.

**Drawing Tips**:
- Use red brush to mark regions
- Use eraser to remove annotations
- Zoom with mouse wheel for precision
""")

                gr.Markdown("### Save Mask")

                save_format = gr.Radio(
                    choices=["npy", "png"], value="npy", label="Mask Format"
                )

                save_mask_btn = gr.Button("Save Mask", variant="primary")
                save_status = gr.Textbox(label="Save Status", interactive=False)
                download_mask_file = gr.File(
                    label="Download mask", interactive=False
                )

            # Right column: Annotation canvas
            with gr.Column(scale=2):
                gr.Markdown("### Annotation Canvas")
                gr.Markdown("**Backdrop image + Mask overlay** (only mask is editable)")

                # Use ImageEditor for mask drawing over backdrop
                annotation_canvas = gr.ImageEditor(
                    label="Draw Mask (backdrop is read-only)",
                    type="pil",
                    image_mode="RGBA",
                    brush=gr.Brush(
                        colors=["#FF0000", "#00FF00", "#0000FF", "#FFFFFF"],
                        default_size=5,
                        default_color="#FF0000",
                    ),
                    eraser=gr.Eraser(default_size=10),
                    sources=[],  # No upload sources - only loaded programmatically
                )

                gr.Markdown("""
**Red brush**: Mark regions (default)
**Green brush**: Alternative marking
**Eraser**: Remove annotations
**Zoom**: Mouse wheel to zoom in/out
""")

        # Comparison section (optional)
        with gr.Accordion("Mask Preview", open=False):
            gr.Markdown("### Current Mask (binary preview)")

            with gr.Row():
                mask_preview = gr.Image(
                    label="Mask Only (extracted from canvas)",
                    type="pil",
                    interactive=False,
                    show_download_button=False,
                    show_share_button=False,
                )

                backdrop_preview = gr.Image(
                    label="Backdrop Only (reference)",
                    type="pil",
                    interactive=False,
                    show_download_button=False,
                    show_share_button=False,
                )

            update_preview_btn = gr.Button("Update Preview")

        # ====================================================================
        # Event Handlers
        # ====================================================================

        def handle_load_npy(npy_file):
            """Load .npy file as backdrop and existing mask if available."""
            backdrop_img, backdrop_arr, mask_arr, info, filename = load_npy_as_backdrop(
                npy_file
            )

            if backdrop_img is None:
                return {
                    backdrop_array_state: None,
                    mask_array_state: None,
                    npy_filename_state: None,
                    annotation_canvas: None,
                    file_info: info,
                }

            # Create composite: backdrop + mask overlay
            composite = create_composite_image(backdrop_img, mask_arr)

            return {
                backdrop_array_state: backdrop_arr,
                mask_array_state: mask_arr,
                npy_filename_state: filename,
                annotation_canvas: composite,
                file_info: info,
            }

        load_btn.click(
            fn=handle_load_npy,
            inputs=[npy_file_input],
            outputs=[
                backdrop_array_state,
                mask_array_state,
                npy_filename_state,
                annotation_canvas,
                file_info,
            ],
        )

        save_mask_btn.click(
            fn=handle_save_mask,
            inputs=[
                annotation_canvas,
                backdrop_array_state,
                npy_filename_state,
                save_format,
            ],
            outputs=[save_status, download_mask_file],
        )

        def handle_update_preview(canvas_output, backdrop_arr, backdrop_img_state):
            """Show just the mask without backdrop and the backdrop separately."""
            if canvas_output is None:
                return None, None

            # Extract mask
            mask_arr = extract_mask_from_canvas(canvas_output, backdrop_arr)
            if mask_arr is None:
                return None, None

            # Convert mask to image
            mask_img = Image.fromarray(mask_arr, mode="L")

            # Show backdrop separately
            backdrop_display = None
            if backdrop_arr is not None:
                backdrop_display = numpy_to_pil(backdrop_arr)

            return mask_img, backdrop_display

        update_preview_btn.click(
            fn=handle_update_preview,
            inputs=[annotation_canvas, backdrop_array_state, backdrop_array_state],
            outputs=[mask_preview, backdrop_preview],
        )

    return tab
