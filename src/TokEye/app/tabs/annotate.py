"""
Mask Annotation Interface Tab for TokEye

This module provides an interface for annotating masks over backdrop images.
The backdrop (loaded from .npy files) is read-only and used as reference only.
Only the mask layer is editable and saved.
"""

import gradio as gr
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Optional, Tuple


def pil_to_numpy(img: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy array."""
    return np.array(img)


def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert numpy array to PIL Image with RGBA support."""
    # Normalize to 0-255 if needed
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        if arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
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


def find_existing_mask(npy_filename: str) -> Optional[str]:
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
) -> Tuple[
    Optional[Image.Image],
    Optional[np.ndarray],
    Optional[np.ndarray],
    str,
    Optional[str],
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
        # Create red overlay
        overlay = Image.new("RGBA", composite.size, (0, 0, 0, 0))

        # Convert mask to PIL
        mask_img = Image.fromarray(mask_arr, mode="L")

        # Create red overlay with alpha channel from mask
        red_overlay = Image.new("RGBA", composite.size)
        for x in range(composite.size[0]):
            for y in range(composite.size[1]):
                if x < mask_arr.shape[1] and y < mask_arr.shape[0]:
                    mask_val = mask_arr[y, x]
                    if mask_val > 0:
                        # Red with semi-transparency
                        red_overlay.putpixel((x, y), (255, 0, 0, 128))
                    else:
                        red_overlay.putpixel((x, y), (0, 0, 0, 0))

        # Composite the overlay
        composite = Image.alpha_composite(composite, red_overlay)

    return composite


def save_mask_annotation(
    mask_arr: Optional[np.ndarray], npy_filename: str, format_choice: str = "npy"
) -> Optional[str]:
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


def extract_mask_from_canvas(
    canvas_output, backdrop_arr: Optional[np.ndarray]
) -> Optional[np.ndarray]:
    """
    Extract only the mask layer from the canvas, removing the backdrop.

    Args:
        canvas_output: Output from gr.ImageEditor
        backdrop_arr: Original backdrop array for comparison

    Returns:
        Binary mask array (0 or 255)
    """
    if canvas_output is None:
        return None

    try:
        # Handle different output formats from ImageEditor
        if isinstance(canvas_output, dict):
            # Try to get composite or background
            composite = canvas_output.get("composite")
            if composite is None:
                composite = canvas_output.get("background")

            if composite is None:
                # Try layers
                layers = canvas_output.get("layers", [])
                if not layers:
                    return None
                composite = layers[0]

            # Convert to array
            if isinstance(composite, Image.Image):
                modified_arr = pil_to_numpy(composite)
            else:
                modified_arr = np.array(composite)
        else:
            # Direct image
            if isinstance(canvas_output, Image.Image):
                modified_arr = pil_to_numpy(canvas_output)
            else:
                modified_arr = np.array(canvas_output)

        # Extract mask by comparing to backdrop or detecting drawn regions
        # Strategy: Look for red channel intensity (user draws in red by default)
        if modified_arr.ndim == 3:
            # Extract red channel
            red_channel = modified_arr[:, :, 0]

            # If we have backdrop, subtract it to isolate drawings
            if backdrop_arr is not None:
                if backdrop_arr.ndim == 3:
                    backdrop_red = backdrop_arr[:, :, 0]
                else:
                    backdrop_red = backdrop_arr

                # Resize if needed
                if backdrop_red.shape != red_channel.shape:
                    from PIL import Image as PILImage

                    backdrop_img = PILImage.fromarray(backdrop_red)
                    backdrop_img = backdrop_img.resize(
                        (red_channel.shape[1], red_channel.shape[0])
                    )
                    backdrop_red = np.array(backdrop_img)

                # Difference
                diff = red_channel.astype(np.int16) - backdrop_red.astype(np.int16)
                mask = (diff > 30).astype(
                    np.uint8
                ) * 255  # Threshold for new annotations
            else:
                # No backdrop - just threshold red channel
                mask = (red_channel > 128).astype(np.uint8) * 255
        else:
            # Grayscale - threshold directly
            mask = (modified_arr > 128).astype(np.uint8) * 255

        return mask

    except Exception as e:
        print(f"Error extracting mask: {e}")
        return None


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

        def handle_save_mask(canvas_output, backdrop_arr, filename, save_fmt):
            """Save only the mask, not the backdrop."""
            if canvas_output is None or filename is None:
                return "Error: No annotation to save"

            try:
                mask_arr = extract_mask_from_canvas(canvas_output, backdrop_arr)
                if mask_arr is None:
                    return "Error: Could not extract mask from canvas"

                filepath = save_mask_annotation(mask_arr, filename, save_fmt)
                if filepath:
                    return f"Mask saved successfully to: {filepath}"
                else:
                    return "Save failed"
            except Exception as e:
                return f"Error: {str(e)}"

        save_mask_btn.click(
            fn=handle_save_mask,
            inputs=[
                annotation_canvas,
                backdrop_array_state,
                npy_filename_state,
                save_format,
            ],
            outputs=[save_status],
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
