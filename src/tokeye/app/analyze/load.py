from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from huggingface_hub import try_to_load_from_cache

from tokeye import hub, inference
from tokeye.examples import make_example_signal
from tokeye.inference import model_infer, signal_to_spectrogram
from tokeye.transforms import (
    DEFAULT_CLIP_DC,
    DEFAULT_CLIP_HIGH,
    DEFAULT_CLIP_LOW,
    DEFAULT_HOP,
    DEFAULT_N_FFT,
    compute_stft,
)

if TYPE_CHECKING:
    import torch.nn as nn

__all__ = [
    "find_models",
    "find_signals",
    "is_model_cached",
    "load_example_signal",
    "load_multi",
    "load_single",
    "model_infer",
    "model_load",
    "signal_load",
]

logger = logging.getLogger(__name__)

MODEL_EXTENSIONS = [".pt", ".pt2"]
MODEL_DIR = Path("model")
SIGNAL_EXTENSIONS = [".npy"]


# Directory Scanning
def find_models() -> list[str]:
    model_dir = MODEL_DIR
    if not model_dir.exists():
        return []
    models = []
    for ext in MODEL_EXTENSIONS:
        models.extend(model_dir.glob(f"*{ext}"))
    return sorted([str(m) for m in models])


def find_signals(
    dir: str | None = None,
) -> list[str]:
    if dir is None:
        return []

    dir_path = Path(dir)
    if not dir_path.exists() or not dir_path.is_dir():
        return []

    signals = []
    for ext in SIGNAL_EXTENSIONS:
        signals.extend(dir_path.glob(f"*{ext}"))
    return sorted([s.name for s in signals])


# Model Functions
def is_model_cached(name: str) -> bool:
    """Check whether a registry model's weights are already in the HF cache."""
    spec = hub.MODEL_REGISTRY[name]
    cached = try_to_load_from_cache(hub.DEFAULT_REPO_ID, spec.filename)
    return isinstance(cached, str)


def model_load(
    source: str | Path,
    device: str = "auto",
) -> nn.Module:
    """Load a model by registry name or local path, then warm it up."""
    logger.info(f"Loading model: {source}")
    model = hub.load_model(source, device)
    inference.warmup(model)
    logger.info("Model ready for inference")
    return model


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


def load_single(
    filepath: Path,
    transform_args: dict,
) -> np.ndarray | None:
    # Load signal
    signal = signal_load(filepath)
    if signal is None:
        logger.error("Failed to load signal")
        return None

    signal_data = np.expand_dims(signal, axis=0)
    logger.info(f"Raw signal shape: {signal.shape}")

    # Apply STFT transform (generalize later)
    n_fft = transform_args.get("n_fft", DEFAULT_N_FFT)
    hop = transform_args.get("hop_length", DEFAULT_HOP)
    clip_dc = transform_args.get("clip_dc", DEFAULT_CLIP_DC)
    clip_low = transform_args.get("percentile_low", DEFAULT_CLIP_LOW)
    clip_high = transform_args.get("percentile_high", DEFAULT_CLIP_HIGH)

    return compute_stft(
        signal_data,
        n_fft=n_fft,
        hop=hop,
        clip_dc=clip_dc,
        clip_low=clip_low,
        clip_high=clip_high,
    )


def load_multi(
    filepaths: list[Path],
    transform_args: dict,
) -> np.ndarray | None:
    # Load both signals
    signal = []
    for filepath in filepaths:
        signal.append(signal_load(filepath))
    if any(s is None for s in signal):
        logger.error("Failed to load one or more signals")
        return None

    signal = np.array(signal)
    logger.info(f"Raw signal shape: {signal.shape}")

    # Apply STFT to both signals (generalize later)
    n_fft = transform_args.get("n_fft", DEFAULT_N_FFT)
    hop = transform_args.get("hop_length", DEFAULT_HOP)
    clip_dc = transform_args.get("clip_dc", DEFAULT_CLIP_DC)
    clip_low = transform_args.get("percentile_low", DEFAULT_CLIP_LOW)
    clip_high = transform_args.get("percentile_high", DEFAULT_CLIP_HIGH)

    return compute_stft(
        signal,
        n_fft=n_fft,
        hop=hop,
        clip_dc=clip_dc,
        clip_low=clip_low,
        clip_high=clip_high,
    )


def load_example_signal(transform_args: dict) -> np.ndarray | None:
    """Generate the deterministic demo signal and compute its spectrogram."""
    signal = make_example_signal()

    n_fft = transform_args.get("n_fft", DEFAULT_N_FFT)
    hop = transform_args.get("hop_length", DEFAULT_HOP)
    clip_dc = transform_args.get("clip_dc", DEFAULT_CLIP_DC)
    clip_low = transform_args.get("percentile_low", DEFAULT_CLIP_LOW)
    clip_high = transform_args.get("percentile_high", DEFAULT_CLIP_HIGH)

    return signal_to_spectrogram(
        signal,
        n_fft=n_fft,
        hop=hop,
        clip_dc=clip_dc,
        clip_low=clip_low,
        clip_high=clip_high,
    )
