"""
Core Analysis Pipeline Tab for TokEye

This module provides the main analysis interface for processing plasma signals
through the complete TokEye pipeline: preprocessing, transform, inference, and visualization.
"""

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Literal
import io
from PIL import Image

# Import TokEye processing utilities
from TokEye.processing import (
    apply_preemphasis,
    compute_stft,
    compute_wavelet,
    tile_spectrogram,
    stitch_predictions,
    load_model,
    batch_inference,
    apply_threshold,
    remove_small_objects,
    create_overlay,
    generate_cache_key,
    CacheManager,
    compute_channel_threshold_bounds,
)

# Initialize global cache manager
cache_manager = CacheManager(cache_dir="cache", max_size_mb=1000, max_entries=500)


def plot_to_image(fig: plt.Figure) -> Image.Image:
    """Convert matplotlib figure to PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    return img


def plot_signal(signal: np.ndarray, title: str = "") -> Image.Image:
    """Plot 1D signal and return as image."""
    fig, ax = plt.subplots(figsize=(10, 4), facecolor="black")
    ax.set_facecolor("black")
    ax.plot(signal, color="#FF6B35", linewidth=0.5)
    ax.axis("off")  # Turn off axes
    plt.tight_layout(pad=0)
    img = plot_to_image(fig)
    plt.close(fig)
    return img


def plot_spectrogram(spec: np.ndarray, title: str = "") -> Image.Image:
    """Plot 2D spectrogram and return as image."""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="black")
    ax.set_facecolor("black")
    im = ax.imshow(spec, aspect="auto", origin="lower", cmap="gist_heat")
    ax.axis("off")  # Turn off axes, title, colorbar
    plt.tight_layout(pad=0)
    img = plot_to_image(fig)
    plt.close(fig)
    return img


# ============================================================================
# Section 1: Input Processing
# ============================================================================


def get_available_signals() -> list:
    """Get list of available .npy signal files in data/."""
    data_dir = Path("data")
    if not data_dir.exists():
        return []
    signals = list(data_dir.glob("*.npy"))
    return [str(s) for s in signals] if signals else []


def load_signal_file(
    file=None, dropdown_path: str = None
) -> Tuple[Optional[np.ndarray], str, Optional[Image.Image]]:
    """
    Load .npy file and display information.
    Accepts either a file upload or a dropdown path selection.

    Returns:
        (signal_array, info_text, plot_image)
    """
    # Determine which source to use
    filepath = None
    if dropdown_path:
        filepath = dropdown_path
    elif file is not None:
        filepath = file.name

    if filepath is None:
        return None, "No file selected or uploaded", None

    try:
        # Load numpy file
        signal = np.load(filepath)

        # Validate signal
        if signal.ndim != 1:
            return None, f"Error: Signal must be 1D, got {signal.ndim}D array", None

        if signal.size == 0:
            return None, "Error: Signal is empty", None

        # Generate info text
        info = f"""
**File Information:**
- Path: {Path(filepath).name}
- Shape: {signal.shape}
- Size: {signal.size:,} samples
- Data type: {signal.dtype}
- Min: {signal.min():.4f}
- Max: {signal.max():.4f}
- Mean: {signal.mean():.4f}
- Std: {signal.std():.4f}
"""

        # Plot signal
        plot_img = plot_signal(signal)

        return signal, info, plot_img

    except Exception as e:
        return None, f"Error loading file: {str(e)}", None


def apply_preemphasis_filter(
    signal: Optional[np.ndarray], enable: bool, alpha: float
) -> Tuple[Optional[np.ndarray], Optional[Image.Image], str]:
    """
    Apply pre-emphasis filter to signal.

    Returns:
        (filtered_signal, plot_image, status_text)
    """
    if signal is None:
        return None, None, "No signal loaded"

    try:
        if enable:
            filtered = apply_preemphasis(signal, alpha=alpha)
            plot_img = plot_signal(filtered)
            status = f"Pre-emphasis applied with alpha={alpha}"
            return filtered, plot_img, status
        else:
            plot_img = plot_signal(signal)
            return signal, plot_img, "Pre-emphasis disabled, using original signal"

    except Exception as e:
        return None, None, f"Error applying pre-emphasis: {str(e)}"


# ============================================================================
# Section 2: Transform Computation
# ============================================================================


def compute_transform(
    signal: Optional[np.ndarray],
    transform_type: str,
    # STFT parameters
    n_fft: int,
    hop_length: int,
    clip_dc: bool,
    # Wavelet parameters
    wavelet_type: str,
    wavelet_level: int,
    wavelet_mode: str,
    # Percentile clipping parameters
    percentile_low: float,
    percentile_high: float,
) -> Tuple[Optional[np.ndarray], Optional[Image.Image], str]:
    """
    Compute STFT or Wavelet transform with caching.

    Returns:
        (transform_array, plot_image, status_text)
    """
    if signal is None:
        return None, None, "No signal loaded"

    try:
        if transform_type == "STFT":
            # Generate cache key
            params = {
                "transform": "stft",
                "n_fft": n_fft,
                "hop_length": hop_length,
                "clip_dc": clip_dc,
                "percentile_low": percentile_low,
                "percentile_high": percentile_high,
            }
            cache_key = generate_cache_key(signal, params, prefix="stft")

            # Check cache
            if cache_manager.exists(cache_key, "spectrogram"):
                result = cache_manager.load(cache_key, "spectrogram")
                status = f"Cache hit! STFT loaded from cache."
            else:
                # Compute STFT
                result = compute_stft(
                    signal,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    clip_dc=clip_dc,
                    percentile_low=percentile_low,
                    percentile_high=percentile_high,
                )
                # Save to cache
                cache_manager.save(cache_key, result, "spectrogram")
                status = f"STFT computed and cached. Shape: {result.shape}"

            plot_img = plot_spectrogram(result)

        else:  # Wavelet
            # Generate cache key
            params = {
                "transform": "wavelet",
                "wavelet": wavelet_type,
                "level": wavelet_level,
                "mode": wavelet_mode,
                "percentile_low": percentile_low,
                "percentile_high": percentile_high,
            }
            cache_key = generate_cache_key(signal, params, prefix="wavelet")

            # Check cache
            if cache_manager.exists(cache_key, "wavelet"):
                result = cache_manager.load(cache_key, "wavelet")
                status = f"Cache hit! Wavelet loaded from cache."
            else:
                # Compute wavelet
                result = compute_wavelet(
                    signal,
                    wavelet=wavelet_type,
                    level=wavelet_level,
                    mode=wavelet_mode,
                    percentile_low=percentile_low,
                    percentile_high=percentile_high,
                )
                # Save to cache
                cache_manager.save(cache_key, result, "wavelet")
                status = f"Wavelet computed and cached. Shape: {result.shape}"

            plot_img = plot_spectrogram(result)

        return result, plot_img, status

    except Exception as e:
        return None, None, f"Error computing transform: {str(e)}"


def save_transform_image(img: Optional[Image.Image]) -> Optional[str]:
    """Save transform visualization to file."""
    if img is None:
        gr.Warning("No image to save")
        return None

    try:
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = output_dir / f"transform_{timestamp}.png"

        img.save(filepath)
        gr.Info(f"Saved to {filepath}")
        return str(filepath)

    except Exception as e:
        gr.Warning(f"Failed to save: {str(e)}")
        return None


# ============================================================================
# Section 3: Model Inference
# ============================================================================


def get_available_models() -> list:
    """Get list of available .pt model files."""
    model_dir = Path("model")
    if not model_dir.exists():
        return []

    models = list(model_dir.glob("*.pt")) + list(model_dir.glob("*.pt2"))
    return [str(m) for m in models] if models else []


def run_inference(
    spectrogram: Optional[np.ndarray],
    model_path: str,
    tile_size: int,
    batch_size: int,
    progress=gr.Progress(),
) -> Tuple[Optional[np.ndarray], str, str]:
    """
    Run model inference on spectrogram.

    Returns:
        (predictions, raw_output_text, status_text)
    """
    if spectrogram is None:
        return None, "", "No spectrogram loaded"

    if not model_path:
        return None, "", "No model selected"

    try:
        progress(0, desc="Loading model...")

        # Generate cache key for inference
        params = {
            "model": model_path,
            "tile_size": tile_size,
            "batch_size": batch_size,
        }
        cache_key = generate_cache_key(spectrogram, params, prefix="inference")

        # Check cache
        if cache_manager.exists(cache_key, "inference"):
            predictions = cache_manager.load(cache_key, "inference")
            status = f"Cache hit! Predictions loaded from cache."

            # Generate output text
            output_text = f"""
**Prediction Statistics:**
- Shape: {predictions.shape}
- Min: {predictions.min():.4f}
- Max: {predictions.max():.4f}
- Mean: {predictions.mean():.4f}
- Non-zero pixels: {np.count_nonzero(predictions):,}
"""
            return predictions, output_text, status

        # Load model
        model = load_model(model_path, device="auto")

        progress(0.2, desc="Tiling spectrogram...")

        # Debug: print spectrogram shape
        print(
            f"DEBUG: Input spectrogram shape: {spectrogram.shape}, ndim: {spectrogram.ndim}"
        )

        # Add channel dimension if needed (C, H, W)
        if spectrogram.ndim == 2:
            spec_with_channels = spectrogram[np.newaxis, :, :]  # Add channel dim
            print(f"DEBUG: Added channel dim, new shape: {spec_with_channels.shape}")
        else:
            spec_with_channels = spectrogram
            print(
                f"DEBUG: Spectrogram already has channels, shape: {spec_with_channels.shape}"
            )

        # Tile spectrogram
        tiles, metadata = tile_spectrogram(spec_with_channels, tile_size=tile_size)
        print(f"DEBUG: Created {len(tiles)} tiles, first tile shape: {tiles[0].shape}")

        progress(0.4, desc=f"Running inference on {len(tiles)} tiles...")

        # Run inference
        predictions_tiles = batch_inference(
            model, tiles, batch_size=batch_size, show_progress=False
        )

        progress(0.8, desc="Stitching predictions...")

        # Update metadata to reflect actual prediction channels
        # (model may output different number of channels than input)
        if len(predictions_tiles) > 0:
            pred_shape = predictions_tiles[0].shape
            if len(pred_shape) == 3:  # (C, H, W)
                metadata["num_channels"] = pred_shape[0]
                metadata["has_channels"] = True
                print(
                    f"DEBUG: Updated metadata for {pred_shape[0]}-channel predictions"
                )
            elif len(pred_shape) == 2:  # (H, W)
                metadata["num_channels"] = None
                metadata["has_channels"] = False
                print(f"DEBUG: Predictions are 2D (no channel dimension)")

        # Stitch predictions back together
        predictions_full = stitch_predictions(
            predictions_tiles, metadata, blend_overlap=True
        )

        # Keep multi-channel predictions for per-channel thresholding
        # No longer combine channels here - will be handled in visualization

        # Save to cache
        cache_manager.save(cache_key, predictions_full, "inference")

        progress(1.0, desc="Complete!")

        # Generate output text
        if predictions_full.ndim == 3:
            num_channels = predictions_full.shape[0]
            channel_info = "\n".join(
                [
                    f"  Channel {i}: min={predictions_full[i].min():.4f}, max={predictions_full[i].max():.4f}, mean={predictions_full[i].mean():.4f}"
                    for i in range(num_channels)
                ]
            )
            output_text = f"""
**Inference Results:**
- Model: {Path(model_path).name}
- Tiles processed: {len(tiles)}
- Prediction shape: {predictions_full.shape} ({num_channels} channels)
{channel_info}
"""
        else:
            output_text = f"""
**Inference Results:**
- Model: {Path(model_path).name}
- Tiles processed: {len(tiles)}
- Prediction shape: {predictions_full.shape}
- Min: {predictions_full.min():.4f}
- Max: {predictions_full.max():.4f}
- Mean: {predictions_full.mean():.4f}
- Non-zero pixels: {np.count_nonzero(predictions_full):,}
"""

        status = f"Inference completed successfully. {len(tiles)} tiles processed."

        return predictions_full, output_text, status

    except Exception as e:
        return None, "", f"Error during inference: {str(e)}"


# ============================================================================
# Section 4: Visualization
# ============================================================================


def compute_threshold_bounds(
    predictions: Optional[np.ndarray],
) -> Tuple[str, Dict[str, Any]]:
    """
    Compute threshold bounds for each channel in predictions.

    Returns:
        (info_text, threshold_bounds_dict)
    """
    if predictions is None:
        return "No predictions available", {}

    try:
        # Handle multi-channel predictions
        if predictions.ndim == 3:
            num_channels = predictions.shape[0]
            bounds = {}
            info_lines = [
                f"**Threshold Bounds (computed for {num_channels} channels):**"
            ]

            for i in range(num_channels):
                lower, upper = compute_channel_threshold_bounds(predictions[i])
                bounds[f"ch{i}"] = {"lower": lower, "upper": upper}
                info_lines.append(
                    f"- Channel {i}: lower={lower:.3f}, upper={upper:.3f}"
                )

            info_text = "\n".join(info_lines)
        else:
            # Single channel
            lower, upper = compute_channel_threshold_bounds(predictions)
            bounds = {"ch0": {"lower": lower, "upper": upper}}
            info_text = (
                f"**Threshold Bounds:**\n- Lower: {lower:.3f}\n- Upper: {upper:.3f}"
            )

        return info_text, bounds

    except Exception as e:
        return f"Error computing bounds: {str(e)}", {}


def generate_visualization(
    spectrogram: Optional[np.ndarray],
    predictions: Optional[np.ndarray],
    ch0_lower: float,
    ch0_upper: float,
    ch1_lower: float,
    ch1_upper: float,
    min_obj_size: int,
    overlay_mode: str,
    overlay_alpha: float,
) -> Tuple[Optional[Image.Image], str]:
    """
    Generate final visualization with per-channel threshold overlay.

    Returns:
        (overlay_image, statistics_text)
    """
    if spectrogram is None or predictions is None:
        return None, "No data available for visualization"

    try:
        import cv2

        # Handle multi-channel predictions
        num_channels = 1
        channel_masks = []

        if predictions.ndim == 3:
            num_channels = predictions.shape[0]

            # Apply thresholds to each channel
            for i in range(num_channels):
                if i == 0:
                    # Channel 0: use pixels between lower and upper threshold
                    mask = (predictions[i] >= ch0_lower) & (predictions[i] <= ch0_upper)
                elif i == 1:
                    # Channel 1: use pixels between lower and upper threshold
                    mask = (predictions[i] >= ch1_lower) & (predictions[i] <= ch1_upper)
                else:
                    # Additional channels: use 0.5 threshold
                    mask = predictions[i] >= 0.5

                channel_masks.append(mask.astype(np.uint8))

            # Combine masks with logical OR
            if num_channels == 1:
                binary_mask = channel_masks[0]
            else:
                binary_mask = np.logical_or.reduce(channel_masks).astype(np.uint8)
        else:
            # Single channel prediction
            binary_mask = (predictions >= ch0_lower) & (predictions <= ch0_upper)
            binary_mask = binary_mask.astype(np.uint8)
            channel_masks = [binary_mask]

        # Remove small objects
        cleaned_mask, num_objects = remove_small_objects(
            binary_mask, min_size=min_obj_size, connectivity=8
        )

        # Create overlay
        # Map UI mode to function mode
        if overlay_mode == "White":
            mode: Literal["white", "bicolor", "hsv"] = "white"
        elif overlay_mode == "Bicolor":
            mode = "bicolor"
        else:
            mode = "hsv"

        # For visualization, we need 2D spectrogram
        if spectrogram.ndim == 3:
            # If spectrogram has channels, use first channel
            spec_2d = spectrogram[0]
        else:
            spec_2d = spectrogram

        overlay_rgb = create_overlay(
            spec_2d,
            cleaned_mask,
            mode=mode,
            alpha=overlay_alpha,
        )

        # Convert to PIL Image and resize for better aspect ratio display
        overlay_img = Image.fromarray(overlay_rgb)

        # Calculate better display size to match Section 2 aspect ratio
        # Keep reasonable dimensions that aren't too wide
        height, width = overlay_rgb.shape[:2]
        aspect_ratio = width / height

        # Target a max width of ~1200 pixels for display
        if aspect_ratio > 2.0:  # Very wide image
            display_width = 1200
            display_height = int(display_width / aspect_ratio)
        else:
            # Use a reasonable size based on height
            display_height = 600
            display_width = int(display_height * aspect_ratio)

        overlay_img = overlay_img.resize(
            (display_width, display_height), Image.Resampling.LANCZOS
        )

        # Compute statistics
        total_pixels = cleaned_mask.size
        detected_pixels = np.count_nonzero(cleaned_mask)
        coverage = detected_pixels / total_pixels * 100

        # Compute per-channel statistics if multi-channel
        if predictions.ndim == 3 and num_channels == 2:
            # Count pixels per channel
            ch0_pixels = np.count_nonzero(channel_masks[0])
            ch1_pixels = np.count_nonzero(channel_masks[1])

            stats_text = f"""
**Detection Statistics:**
- Total objects detected: {num_objects}
- Channel 0 pixels: {ch0_pixels:,} ({ch0_pixels / total_pixels * 100:.2f}%)
- Channel 1 pixels: {ch1_pixels:,} ({ch1_pixels / total_pixels * 100:.2f}%)
- Total area coverage: {coverage:.2f}%
- Thresholds: Ch0=[{ch0_lower:.3f}, {ch0_upper:.3f}], Ch1=[{ch1_lower:.3f}, {ch1_upper:.3f}]
- Min object size: {min_obj_size} pixels
"""
        else:
            # Simple heuristic to estimate coherent vs transient
            if num_objects > 0:
                mask_uint8 = (cleaned_mask > 0).astype(np.uint8) * 255
                _, _, stats_array, _ = cv2.connectedComponentsWithStats(
                    mask_uint8, connectivity=8
                )

                areas = stats_array[1:, cv2.CC_STAT_AREA]  # Skip background
                coherent_count = np.sum(areas > 100)  # Arbitrary threshold
                transient_count = num_objects - coherent_count
            else:
                coherent_count = 0
                transient_count = 0

            stats_text = f"""
**Detection Statistics:**
- Total objects detected: {num_objects}
- Estimated coherent structures: {coherent_count}
- Estimated transient events: {transient_count}
- Total area coverage: {coverage:.2f}%
- Threshold range: [{ch0_lower:.3f}, {ch0_upper:.3f}]
- Min object size: {min_obj_size} pixels
"""

        return overlay_img, stats_text

    except Exception as e:
        return None, f"Error generating visualization: {str(e)}"


def save_result_image(img: Optional[Image.Image]) -> Optional[str]:
    """Save final result visualization to file."""
    if img is None:
        gr.Warning("No image to save")
        return None

    try:
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = output_dir / f"result_{timestamp}.png"

        img.save(filepath)
        gr.Info(f"Saved to {filepath}")
        return str(filepath)

    except Exception as e:
        gr.Warning(f"Failed to save: {str(e)}")
        return None


# ============================================================================
# Gradio Interface
# ============================================================================


def analyze_tab():
    """Create the analysis tab interface."""

    with gr.Column() as tab:
        gr.Markdown("# TokEye Analysis Pipeline")
        gr.Markdown(
            "Upload a signal file (.npy) and process through the complete analysis pipeline."
        )

        # State variables
        signal_state = gr.State(None)
        processed_signal_state = gr.State(None)
        transform_state = gr.State(None)
        predictions_state = gr.State(None)

        # ====================================================================
        # Section 1: Input
        # ====================================================================
        with gr.Accordion("1. Input Signal", open=True):
            with gr.Row():
                with gr.Column(scale=1):
                    pass  # Empty column for centering

                with gr.Column(scale=2):
                    with gr.Row():
                        signal_dropdown = gr.Dropdown(
                            choices=get_available_signals(),
                            label="Select Signal from data/",
                            allow_custom_value=False,
                        )
                        refresh_signals_btn = gr.Button("🔄 Refresh")

                    gr.Markdown("Or upload a custom file:")
                    file_input = gr.File(
                        label="Upload Signal File (.npy)", file_types=[".npy"]
                    )
                    upload_btn = gr.Button("Load File", variant="primary")

                with gr.Column(scale=1):
                    pass  # Empty column for centering

            with gr.Row():
                file_info = gr.Markdown("*No file loaded*")

            signal_plot = gr.Image(
                label="Signal Visualization",
                type="pil",
                interactive=False,
                show_download_button=False,
                show_share_button=False,
                sources=[],
            )

            gr.Markdown("### Pre-emphasis Filter")
            with gr.Row():
                preemph_enable = gr.Checkbox(label="Enable Pre-emphasis", value=True)
                preemph_alpha = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.97,
                    step=0.01,
                    label="Alpha coefficient",
                )

            apply_preemph_btn = gr.Button("Apply Pre-emphasis")
            preemph_status = gr.Textbox(label="Status", interactive=False)
            filtered_plot = gr.Image(
                label="Processed Signal",
                type="pil",
                interactive=False,
                show_download_button=False,
                show_share_button=False,
                sources=[],
            )

        # ====================================================================
        # Section 2: Transform
        # ====================================================================
        with gr.Accordion("2. Time-Frequency Transform", open=False):
            transform_type = gr.Radio(
                choices=["STFT", "Wavelet"], value="STFT", label="Transform Type"
            )

            # STFT parameters
            with gr.Group(visible=True) as stft_params:
                gr.Markdown("### STFT Parameters")
                with gr.Row():
                    n_fft = gr.Slider(
                        minimum=256, maximum=4096, value=1024, step=256, label="N_FFT"
                    )
                    hop_length = gr.Slider(
                        minimum=64, maximum=512, value=128, step=64, label="Hop Length"
                    )
                clip_dc = gr.Checkbox(label="Clip DC Bin", value=True)

            # Wavelet parameters
            with gr.Group(visible=False) as wavelet_params:
                gr.Markdown("### Wavelet Parameters")
                with gr.Row():
                    wavelet_type = gr.Dropdown(
                        choices=[f"db{i}" for i in range(1, 21)],
                        value="db8",
                        label="Wavelet Type",
                    )
                    wavelet_level = gr.Slider(
                        minimum=1,
                        maximum=12,
                        value=9,
                        step=1,
                        label="Decomposition Level",
                    )
                wavelet_mode = gr.Dropdown(
                    choices=[
                        "sym",
                        "periodic",
                        "zero",
                        "constant",
                        "smooth",
                        "reflect",
                    ],
                    value="sym",
                    label="Extension Mode",
                )

            # Percentile clipping parameters
            gr.Markdown("### Percentile Clipping")
            with gr.Row():
                percentile_low = gr.Slider(
                    minimum=0.0,
                    maximum=10.0,
                    value=1.0,
                    step=0.1,
                    label="Lower Percentile Clip",
                )
                percentile_high = gr.Slider(
                    minimum=90.0,
                    maximum=100.0,
                    value=99.0,
                    step=0.1,
                    label="Upper Percentile Clip",
                )

            compute_transform_btn = gr.Button("Compute Transform", variant="primary")

            transform_plot = gr.Image(
                label="Transform Visualization",
                type="pil",
                interactive=False,
                show_download_button=False,
                show_share_button=False,
                sources=[],
            )
            transform_status = gr.Textbox(label="Status", interactive=False)

            save_transform_btn = gr.Button("Save Analysis Image")

        # ====================================================================
        # Section 3: Inference
        # ====================================================================
        with gr.Accordion("3. Model Inference", open=False):
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=get_available_models(),
                    label="Select Model",
                    allow_custom_value=True,
                )
                refresh_models_btn = gr.Button("= Refresh")

            with gr.Row():
                tile_size_slider = gr.Slider(
                    minimum=64, maximum=1024, value=512, step=64, label="Tile Size"
                )
                batch_size_slider = gr.Slider(
                    minimum=1, maximum=64, value=32, step=1, label="Batch Size"
                )

            run_inference_btn = gr.Button("Run Inference", variant="primary")

            inference_output = gr.Textbox(
                label="Inference Results", lines=8, interactive=False
            )
            inference_status = gr.Textbox(label="Status", interactive=False)

        # ====================================================================
        # Section 4: Visualization
        # ====================================================================
        with gr.Accordion("4. Visualization", open=False):
            # Compute threshold bounds button
            compute_bounds_btn = gr.Button(
                "Compute Threshold Bounds", variant="secondary"
            )
            threshold_bounds_info = gr.Textbox(
                label="Threshold Bounds",
                lines=4,
                interactive=False,
                value="Click 'Compute Threshold Bounds' after inference",
            )

            # Channel 0 thresholds
            gr.Markdown("### Channel 0 Thresholds")
            with gr.Row():
                ch0_lower_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0,
                    step=0.01,
                    label="Channel 0 Lower Threshold",
                )
                ch0_upper_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=1.0,
                    step=0.01,
                    label="Channel 0 Upper Threshold",
                )

            # Channel 1 thresholds
            gr.Markdown("### Channel 1 Thresholds")
            with gr.Row():
                ch1_lower_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0,
                    step=0.01,
                    label="Channel 1 Lower Threshold",
                )
                ch1_upper_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=1.0,
                    step=0.01,
                    label="Channel 1 Upper Threshold",
                )

            # Other parameters
            gr.Markdown("### Post-Processing & Display")
            with gr.Row():
                min_size_slider = gr.Slider(
                    minimum=10,
                    maximum=500,
                    value=50,
                    step=10,
                    label="Min Object Size (pixels)",
                )

            with gr.Row():
                overlay_mode_radio = gr.Radio(
                    choices=["White", "Bicolor", "HSV"],
                    value="White",
                    label="Overlay Mode",
                )
                overlay_alpha_slider = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.6, step=0.1, label="Overlay Alpha"
                )

            generate_viz_btn = gr.Button("Generate Visualization", variant="primary")

            viz_output = gr.Image(
                label="Final Visualization",
                type="pil",
                interactive=False,
                show_download_button=False,
                show_share_button=False,
                sources=[],
            )
            viz_stats = gr.Textbox(
                label="Detection Statistics", lines=8, interactive=False
            )

            save_result_btn = gr.Button("Save Result Image")

        # ====================================================================
        # Event Handlers
        # ====================================================================

        # Upload file
        upload_btn.click(
            fn=load_signal_file,
            inputs=[file_input, signal_dropdown],
            outputs=[signal_state, file_info, signal_plot],
        )

        # Dropdown signal selection
        signal_dropdown.change(
            fn=load_signal_file,
            inputs=[file_input, signal_dropdown],
            outputs=[signal_state, file_info, signal_plot],
        )

        # Refresh signals
        refresh_signals_btn.click(
            fn=lambda: gr.update(choices=get_available_signals()),
            inputs=[],
            outputs=[signal_dropdown],
        )

        # Apply pre-emphasis
        apply_preemph_btn.click(
            fn=apply_preemphasis_filter,
            inputs=[signal_state, preemph_enable, preemph_alpha],
            outputs=[processed_signal_state, filtered_plot, preemph_status],
        )

        # Toggle transform parameters visibility
        def toggle_transform_params(choice):
            return {
                stft_params: gr.update(visible=(choice == "STFT")),
                wavelet_params: gr.update(visible=(choice == "Wavelet")),
            }

        transform_type.change(
            fn=toggle_transform_params,
            inputs=[transform_type],
            outputs=[stft_params, wavelet_params],
        )

        # Compute transform
        compute_transform_btn.click(
            fn=compute_transform,
            inputs=[
                processed_signal_state,
                transform_type,
                n_fft,
                hop_length,
                clip_dc,
                wavelet_type,
                wavelet_level,
                wavelet_mode,
                percentile_low,
                percentile_high,
            ],
            outputs=[transform_state, transform_plot, transform_status],
        )

        # Save transform image
        save_transform_btn.click(
            fn=save_transform_image, inputs=[transform_plot], outputs=[]
        )

        # Refresh models
        refresh_models_btn.click(
            fn=lambda: gr.update(choices=get_available_models()),
            inputs=[],
            outputs=[model_dropdown],
        )

        # Run inference
        run_inference_btn.click(
            fn=run_inference,
            inputs=[
                transform_state,
                model_dropdown,
                tile_size_slider,
                batch_size_slider,
            ],
            outputs=[predictions_state, inference_output, inference_status],
        )

        # Compute threshold bounds
        compute_bounds_btn.click(
            fn=compute_threshold_bounds,
            inputs=[predictions_state],
            outputs=[threshold_bounds_info],
        )

        # Generate visualization
        generate_viz_btn.click(
            fn=generate_visualization,
            inputs=[
                transform_state,
                predictions_state,
                ch0_lower_slider,
                ch0_upper_slider,
                ch1_lower_slider,
                ch1_upper_slider,
                min_size_slider,
                overlay_mode_radio,
                overlay_alpha_slider,
            ],
            outputs=[viz_output, viz_stats],
        )

        # Save result image
        save_result_btn.click(fn=save_result_image, inputs=[viz_output], outputs=[])

    return tab
