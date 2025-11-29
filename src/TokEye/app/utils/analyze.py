import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from skimage import measure
from tqdm.auto import tqdm

from TokEye.processing.postprocess import apply_threshold, remove_small_objects
from TokEye.processing.transforms import compute_stft

logger = logging.getLogger(__name__)


# Signal Functions
def signal_load(file=None, dropdown_path: str = "") -> np.ndarray | None:
    filepath = None
    if dropdown_path:
        filepath = dropdown_path
    elif file is not None:
        filepath = file.name
    else:
        logger.error("No file selected or uploaded")
        return None

    try:
        signal = np.load(filepath)
    except Exception as e:
        logger.error(f"Failed to load signal: {e}")
        return None
    if signal.ndim != 1:
        logger.error("Signal must be 1D array")
        return None
    if signal.size == 0:
        logger.error("Signal is empty")
        return None
    return signal


# Model Functions
def model_load(
    model_path: str | Path,
    device: str = "auto",
) -> nn.Module | torch.export.ExportedProgram:
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        if model_path.suffix == ".pt2":
            module = torch.export.load(str(model_path))
            model = module.module()
        else:
            model = torch.jit.load(
                str(model_path),
                map_location=device,
            )
            model.eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}") from e

    return model


def model_info(model: nn.Module) -> dict:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    nparams = sum(p.numel() for p in model.parameters())

    return {
        "device": str(device) if device else "unknown",
        "dtype": str(dtype) if dtype else "unknown",
        "nparams": nparams,
    }


def model_warmup(
    model: nn.Module,
    input_shape: tuple,
    num_iterations: int = 5,
    dtype: torch.dtype = torch.float32,
):
    logger.info("Warming up model...")
    device = next(model.parameters()).device
    dummy_input = torch.randn(*input_shape, device=device, dtype=dtype)
    with torch.no_grad():
        for _ in tqdm(range(num_iterations)):
            _ = model(dummy_input)
    logger.info("Model warmup complete.")


def model_infer(
    inp: np.ndarray | None,
    model: nn.Module,
    mode: str = "single",
) -> np.ndarray | None:
    out = None
    if mode == "single":
        out = torch.from_numpy(inp)
        out = out.unsqueeze(0).unsqueeze(0)
        out = model(out)
        out = out.squeeze(0).squeeze(0)
        out = out.numpy()
    else:
        logger.error(f"Unsupported inference mode: {mode}")
    return out


# Directory Scanning
def get_available_models() -> list[str]:
    """Scan model/ for .pt and .pt2 files."""
    model_dir = Path("model")
    if not model_dir.exists():
        return []
    models = list(model_dir.glob("*.pt")) + list(model_dir.glob("*.pt2"))
    return [str(m) for m in models] if models else []


def get_available_signals() -> list[str]:
    """Scan data/ for .npy files."""
    data_dir = Path("data")
    if not data_dir.exists():
        return []
    signals = list(data_dir.glob("*.npy"))
    return [str(s) for s in signals] if signals else []


# Pipeline Handlers
def handle_load(model_path: str, signal_path: str) -> tuple:
    """Load model and signal, warmup model."""
    try:
        if not model_path or not signal_path:
            return None, None, "Please select both model and signal"

        model = model_load(model_path)
        model_warmup(model, input_shape=(1, 1, 512, 512), num_iterations=3)
        signal = signal_load(dropdown_path=signal_path)

        if signal is None:
            return None, None, "Failed to load signal"

        status = f"Loaded: {Path(model_path).name} | Signal: {len(signal)} samples"
        return model, signal, status
    except Exception as e:
        logger.error(f"Error in handle_load: {e}")
        return None, None, f"Error: {str(e)}"


def swap_signal(new_path: str) -> tuple:
    """Load new signal without reloading model."""
    try:
        if not new_path:
            return None, "No signal selected"
        signal = signal_load(dropdown_path=new_path)
        if signal is None:
            return None, "Failed to load signal"
        status = f"Signal: {len(signal)} samples"
        return signal, status
    except Exception as e:
        logger.error(f"Error in swap_signal: {e}")
        return None, f"Error: {str(e)}"


def compute_stft_pipeline(
    signal: np.ndarray,
    n_fft: int,
    hop_length: int,
    clip_dc: bool,
    percentile_low: float,
    percentile_high: float,
) -> tuple:
    """Compute STFT and generate preview."""
    try:
        if signal is None:
            return None, None

        spec = compute_stft(
            signal,
            n_fft=n_fft,
            hop_length=hop_length,
            clip_dc=clip_dc,
            percentile_low=percentile_low,
            percentile_high=percentile_high,
        )

        # Generate preview
        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(spec, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()

        # Convert to PIL
        fig.canvas.draw()
        img = Image.frombytes(
            "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
        )
        plt.close(fig)

        return spec, img
    except Exception as e:
        logger.error(f"Error in compute_stft_pipeline: {e}")
        return None, None


def run_inference(spectrogram: np.ndarray, model: nn.Module) -> tuple:
    """Run inference and apply sigmoid."""
    try:
        if spectrogram is None or model is None:
            return None, "Spectrogram or model not loaded"

        inp = torch.from_numpy(spectrogram).unsqueeze(0).unsqueeze(0).float()
        device = next(model.parameters()).device
        inp = inp.to(device)

        with torch.no_grad():
            out = model(inp)
            out = torch.sigmoid(out)

        out = out.squeeze(0).cpu().numpy()

        if out.shape[0] != 2:
            return None, f"Expected 2 channels, got {out.shape[0]}"

        status = f"Inference complete: {out.shape}"
        return out, status
    except Exception as e:
        logger.error(f"Error in run_inference: {e}")
        return None, f"Error: {str(e)}"


# Visualization Renderers
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
    img = Image.frombytes(
        "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
    )
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


def render_labels(
    inference: np.ndarray,
    ch0_enabled: bool,
    ch1_enabled: bool,
    threshold: float,
    min_size: int,
) -> Image.Image:
    """Render labeled components."""
    if inference is None:
        return None

    # Apply threshold and remove small objects
    mask_ch0 = apply_threshold(inference[0], threshold, binary=True)
    mask_ch1 = apply_threshold(inference[1], threshold, binary=True)

    mask_ch0, _ = remove_small_objects(mask_ch0.astype(np.uint8), min_size=min_size)
    mask_ch1, _ = remove_small_objects(mask_ch1.astype(np.uint8), min_size=min_size)

    # Label components
    labels_ch0 = measure.label(mask_ch0, connectivity=2)
    labels_ch1 = measure.label(mask_ch1, connectivity=2)

    # Create RGB image with unique colors
    rgb = np.zeros((*mask_ch0.shape, 3))

    if ch0_enabled and ch1_enabled:
        rgb[:, :, 1] = (labels_ch0 > 0).astype(float)  # Green
        rgb[:, :, 0] = (labels_ch1 > 0).astype(float)  # Red
    elif ch0_enabled:
        rgb[:, :, 1] = (labels_ch0 > 0).astype(float)  # Green
    elif ch1_enabled:
        rgb[:, :, 0] = (labels_ch1 > 0).astype(float)  # Red

    rgb = (rgb * 255).astype(np.uint8)
    return Image.fromarray(rgb)


def update_visualization(
    view_mode: str,
    spectrogram: np.ndarray,
    inference: np.ndarray,
    ch0_enh: bool,
    ch1_enh: bool,
    clip_min: float,
    clip_max: float,
    ch0_mask: bool,
    ch1_mask: bool,
    threshold_mask: float,
    ch0_labels: bool,
    ch1_labels: bool,
    threshold_labels: float,
    min_size: int,
) -> Image.Image:
    """Route to appropriate renderer based on view mode."""
    try:
        if view_mode == "Original":
            return render_original(spectrogram)
        elif view_mode == "Enhanced":
            return render_enhanced(inference, ch0_enh, ch1_enh, clip_min, clip_max)
        elif view_mode == "Mask":
            return render_mask(inference, ch0_mask, ch1_mask, threshold_mask)
        elif view_mode == "Labels":
            return render_labels(
                inference, ch0_labels, ch1_labels, threshold_labels, min_size
            )
        return None
    except Exception as e:
        logger.error(f"Error in update_visualization: {e}")
        return None
