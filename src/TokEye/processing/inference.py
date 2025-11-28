"""
Model Inference Utilities

This module provides utilities for loading PyTorch models and running
batch inference on tiled spectrograms.
"""

from pathlib import Path
from venv import logger

import torch
import torch.nn as nn
from tqdm.auto import tqdm


def load_model(
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


def get_model_info(model: nn.Module) -> dict:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    nparams = sum(p.numel() for p in model.parameters())

    return {
        "device": str(device) if device else "unknown",
        "dtype": str(dtype) if dtype else "unknown",
        "nparams": nparams,
    }


def warmup(
    model: nn.Module,
    input_shape: tuple,
    num_iterations: int = 5,
    dtype: torch.dtype = torch.float32,
):
    logger.info("Warming up model...")
    dummy_input = torch.randn(*input_shape, device=model.device, dtype=dtype)
    with torch.no_grad():
        for _ in tqdm(range(num_iterations)):
            _ = model(dummy_input)
