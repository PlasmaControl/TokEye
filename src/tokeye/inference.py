from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .transforms import compute_stft

logger = logging.getLogger(__name__)

WARMUP_INPUT_SHAPE = (1, 1, 512, 512)  # (batch_size, channels, height, width)


def model_infer(
    inp_array: np.ndarray | None,
    model: nn.Module | torch.export.ExportedProgram | None,
) -> np.ndarray | None:
    if inp_array is None or model is None:
        logger.warning("Missing input or model for inference")
        return None

    logger.info(f"Running inference on input shape: {inp_array.shape}")

    device = next(model.parameters()).device
    inp_array = (inp_array - inp_array.mean()) / (inp_array.std() + 1e-6)
    inp_tensor = torch.from_numpy(inp_array)
    inp_tensor = inp_tensor.unsqueeze(0).unsqueeze(0).float()
    inp_tensor = inp_tensor.to(device)

    with torch.no_grad():
        out_tensor = model(inp_tensor)
    out_tensor = out_tensor[0]

    out_tensor = torch.sigmoid(out_tensor)
    out_tensor = out_tensor.squeeze(0).squeeze(0).cpu()
    return out_tensor.numpy()


def signal_to_spectrogram(signal: np.ndarray, **stft_kwargs) -> np.ndarray:
    """Expand a 1D signal to (1, N) and run it through ``compute_stft``."""
    signal_data = np.expand_dims(signal, axis=0)
    return compute_stft(signal_data, **stft_kwargs)


def warmup(model: nn.Module, iterations: int = 10) -> None:
    """Run dummy forward passes to trigger lazy init / kernel autotuning."""
    device = next(model.parameters()).device
    dummy_input = torch.randn(*WARMUP_INPUT_SHAPE, device=device, dtype=torch.float32)
    with torch.no_grad():
        for _ in tqdm(range(iterations)):
            _ = model(dummy_input)
