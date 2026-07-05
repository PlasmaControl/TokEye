"""One-import Python API: ``from tokeye import TokEye``.

Gradio-free, so it is safe to embed in headless programs. ``TokEye()``
loads the default model (auto-downloading it from Hugging Face on first
use) and the instance is directly callable on raw data::

    from tokeye import TokEye

    eye = TokEye()
    mask = eye(signal)  # 1D time series or 2D spectrogram -> (2, H, W)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import hub
from .inference import model_infer, signal_to_spectrogram
from .transforms import (
    DEFAULT_CLIP_DC,
    DEFAULT_CLIP_HIGH,
    DEFAULT_CLIP_LOW,
    DEFAULT_HOP,
    DEFAULT_N_FFT,
    log_scale,
)

if TYPE_CHECKING:
    from pathlib import Path


class TokEye:
    """Callable wrapper around a loaded TokEye model.

    All arguments have working defaults; ``TokEye()`` is fully configured.
    ``model`` accepts a registry name (downloaded and cached automatically)
    or a path to a local ``.pt``/``.pt2`` checkpoint. The STFT arguments
    only apply to 1D inputs.

    ``log`` applies ``np.log1p`` to 2D spectrogram inputs before inference,
    for spectrograms stored in linear scale (the model expects log-scaled
    input). Off by default; 1D inputs are always log-scaled as part of the
    STFT, so ``log`` is ignored for them.
    """

    def __init__(
        self,
        model: str | Path = hub.DEFAULT_MODEL,
        device: str = "auto",
        n_fft: int = DEFAULT_N_FFT,
        hop: int = DEFAULT_HOP,
        clip_dc: bool = DEFAULT_CLIP_DC,
        clip_low: float = DEFAULT_CLIP_LOW,
        clip_high: float = DEFAULT_CLIP_HIGH,
        log: bool = False,
    ) -> None:
        self.model = hub.load_model(model, device)
        self.log = log
        self._stft_kwargs = {
            "n_fft": n_fft,
            "hop": hop,
            "clip_dc": clip_dc,
            "clip_low": clip_low,
            "clip_high": clip_high,
        }

    def spectrogram(self, data: np.ndarray, log: bool | None = None) -> np.ndarray:
        """Convert input data to the spectrogram the model will see.

        1D input is run through the STFT (which log-scales internally);
        2D input is used as-is, log1p-scaled first when ``log`` is on.
        ``log=None`` defers to the instance-level setting.
        """
        arr = np.asarray(data, dtype=float)
        if arr.ndim == 1:
            return signal_to_spectrogram(arr, **self._stft_kwargs)
        if arr.ndim == 2:
            apply_log = self.log if log is None else log
            return log_scale(arr) if apply_log else arr
        raise ValueError(f"expected 1D signal or 2D spectrogram, got ndim={arr.ndim}")

    def predict(self, data: np.ndarray, log: bool | None = None) -> np.ndarray:
        """Run inference; returns a float mask of shape ``(2, H, W)``.

        Channel 0 = coherent activity, channel 1 = transient activity,
        both sigmoid scores in ``[0, 1]``. Standardization (mean 0, std 1)
        is applied internally — no preprocessing needed.
        """
        return model_infer(self.spectrogram(data, log=log), self.model)

    def __call__(self, data: np.ndarray, log: bool | None = None) -> np.ndarray:
        return self.predict(data, log=log)
