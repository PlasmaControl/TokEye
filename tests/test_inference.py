from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from tokeye.inference import model_infer, signal_to_spectrogram, warmup


class _IdentityLikeModel(nn.Module):
    """Returns a 1-tuple of (B, 2, H, W), mimicking BigTFUNetModel's output."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return (self.conv(x),)


def test_model_infer_output_shape_and_range():
    model = _IdentityLikeModel()
    model.eval()
    inp = np.random.default_rng(0).normal(size=(64, 64)).astype(np.float32)

    out = model_infer(inp, model)

    assert out.shape == (2, 64, 64)
    assert np.all(out >= 0.0) and np.all(out <= 1.0)  # sigmoid range


def test_model_infer_none_input_returns_none():
    model = _IdentityLikeModel()
    assert model_infer(None, model) is None


def test_model_infer_none_model_returns_none():
    inp = np.zeros((32, 32), dtype=np.float32)
    assert model_infer(inp, None) is None


def test_signal_to_spectrogram_returns_2d():
    signal = np.random.default_rng(0).normal(size=4096).astype(np.float64)
    spec = signal_to_spectrogram(signal, n_fft=256, hop=64)
    assert spec.ndim == 2


def test_warmup_runs_without_error():
    model = _IdentityLikeModel()
    model.eval()
    warmup(model, iterations=2)  # should not raise
