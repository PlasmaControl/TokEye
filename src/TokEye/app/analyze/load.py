import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from TokEye.processing.transforms import compute_stft

logger = logging.getLogger(__name__)

DUMMY_INPUT_SHAPE = (1, 1, 512, 512)  # (batch_size, channels, height, width)
WARMUP_ITERATIONS = 10


# Signal Functions
def signal_load(filepath) -> np.ndarray | None:
    try:
        signal = np.load(Path(filepath))
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

    print(f"Loading model: {model_path.name}")
    print(f"Device: {device}")

    try:
        if model_path.suffix == ".ckpt":
            # TODO: Generalize later
            from TokEye.autoprocess.multichannel.step_6d_final import (
                Module,
                default_settings,
            )

            model = Module(num_layers=5, first_layer_size=32, settings=default_settings)
            model = Module.load_from_checkpoint(
                model_path,
                num_layers=5,
                first_layer_size=32,
            )
            model.to(device)
            model.eval()
        elif model_path.suffix == ".pt2":
            module = torch.export.load(str(model_path))
            model = module.module()
            model.to(device)
            print("Model loaded (PyTorch 2.0 export format)")
        else:
            model = torch.jit.load(
                str(model_path),
                map_location=device,
            )
            model.eval()
            print("Model loaded (TorchScript format)")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}") from e

    print(f"Warming up model ({WARMUP_ITERATIONS} iterations)...")
    logger.info(f"Warming up model on device: {device}")
    try:
        dummy_input = torch.randn(
            *DUMMY_INPUT_SHAPE,
            device=device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            for _ in tqdm(range(WARMUP_ITERATIONS)):
                _ = model(dummy_input)
    except Exception as e:
        raise RuntimeError(f"Model warm-up failed: {e}") from e
    print("Model ready for inference")
    return model


def model_infer(
    inp: np.ndarray | None,
    model: nn.Module | torch.export.ExportedProgram | None,
) -> np.ndarray | None:
    if inp is None or model is None:
        print("Missing input or model for inference")
        return None

    print(f"Running inference on input shape: {inp.shape}")

    try:
        device = next(model.parameters()).device
        inp = (inp - inp.mean()) / inp.std()
        inp_tensor = torch.from_numpy(inp)
        inp_tensor = inp_tensor.unsqueeze(0).unsqueeze(0).float()
        inp_tensor = inp_tensor.to(device)

        with torch.no_grad():
            out_tensor = model(inp_tensor)

        out_tensor = out_tensor[0]
        out_tensor = torch.sigmoid(out_tensor)
        out_tensor = out_tensor.squeeze(0).squeeze(0).cpu()
        out = out_tensor.numpy()

        print(f"Inference complete: output shape {out.shape}")
        return out
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        print(f"Error during inference: {e}")
        return None


# Directory Scanning
def get_available_models() -> list[str]:
    """Scan model/ for .pt and .pt2 files."""
    model_dir = Path("model")
    if not model_dir.exists():
        return []
    models = (
        list(model_dir.glob("*.pt"))
        + list(model_dir.glob("*.pt2"))
        + list(model_dir.glob("*.ckpt"))
    )
    return [str(m) for m in models] if models else []


def get_available_signals() -> list[str]:
    """Scan data/ for .npy files."""
    data_dir = Path("data")
    if not data_dir.exists():
        return []
    signals = list(data_dir.glob("*.npy"))
    return [str(s) for s in signals] if signals else []


def load_single(filepath: str, transform_args: dict) -> np.ndarray | None:
    """Load single signal and apply STFT transform."""
    if not filepath or not transform_args:
        logger.error("File path or transform args missing")
        logger.info(f"filepath: {filepath}, transform_args: {transform_args}")
        print("Missing file path or transform arguments")
        return None

    try:
        print(f"Loading signal: {Path(filepath).name}")

        # Load signal
        signal = signal_load(filepath)
        print(signal.shape)
        if signal is None:
            logger.error("Failed to load signal")
            print("Failed to load signal")
            return None
        signal = np.expand_dims(signal, axis=0)  # Add channel dimension
        if signal is None:
            logger.error("Signal is empty")
            print("Failed to load signal")
            return None

        print(f"Signal loaded: {len(signal):,} samples")

        # Apply STFT transform
        n_fft = transform_args.get("n_fft", 1024)
        hop = transform_args.get("hop_length", 256)
        clip_dc = transform_args.get("clip_dc", True)
        clip_low = transform_args.get("percentile_low", 1.0)
        clip_high = transform_args.get("percentile_high", 99.0)

        print(
            f"n_fft={n_fft}\n"
            f"hop_length={hop}"
            f"clip_dc={clip_dc}"
            f"percentiles=[{clip_low}, {clip_high}]"
        )

        spec = compute_stft(
            signal,
            n_fft=n_fft,
            hop=hop,
            clip_dc=clip_dc,
            clip_low=clip_low,
            clip_high=clip_high,
        )

        print(f"Transform complete: shape {spec.shape}")
        logger.info(f"Loaded signal: {filepath}, shape: {spec.shape}")
        return spec
    except Exception as e:
        logger.error(f"Failed to load single signal: {e}")
        print(f"Error: {e}")
        return None


def load_multi(
    signal_1_path: str,
    signal_2_path: str,
    transform_args: dict,
) -> np.ndarray | None:
    """Load two signals and apply STFT transform for cross-signal analysis."""
    if not signal_1_path or not signal_2_path or not transform_args:
        logger.error("Signal paths or transform args missing")
        print("Missing signal paths or transform arguments")
        return None

    try:
        print(
            f"Loading cross-signal: {Path(signal_1_path).name} & {Path(signal_2_path).name}"
        )

        # Load both signals
        signal_1 = signal_load(signal_1_path)
        signal_2 = signal_load(signal_2_path)

        if signal_1 is None or signal_2 is None:
            logger.error("Failed to load one or both signals")
            print("Failed to load one or both signals")
            return None

        signal = np.stack([signal_1, signal_2], axis=0)
        print(f"Signals loaded: {len(signal_1):,} & {len(signal_2):,} samples")

        # Apply STFT to both signals
        n_fft = transform_args.get("n_fft", 1024)
        hop = transform_args.get("hop_length", 256)
        clip_dc = transform_args.get("clip_dc", True)
        clip_low = transform_args.get("percentile_low", 1.0)
        clip_high = transform_args.get("percentile_high", 99.0)

        print(
            f"n_fft={n_fft}\n"
            f"hop_length={hop}"
            f"clip_dc={clip_dc}"
            f"percentiles=[{clip_low}, {clip_high}]"
        )

        spec = compute_stft(
            signal,
            n_fft=n_fft,
            hop=hop,
            clip_dc=clip_dc,
            clip_low=clip_low,
            clip_high=clip_high,
        )
        return spec
    except Exception as e:
        logger.error(f"Failed to load multi signals: {e}")
        print(f"Error: {e}")
        return None
