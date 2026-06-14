"""Step 2a: Create spectrograms from time-series using STFT.

Reads ``(C, T)`` arrays from step_1a (or step_0c) HDF5, applies
``torch.stft`` with configurable ``nfft`` / ``hop_length``, writes
``(C, F, T_spec, 2)`` (real/imag) to the step_2a HDF5.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from .utils.configuration import load_settings
from .utils.hdf5_io import (
    create_step_file,
    get_sample_count,
    iter_samples,
    write_sample,
)

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

default_settings = {
    "nfft": 1024,
    "hop_length": 128,
    "input_h5": Path("data/cache/step_1a.h5"),
    "output_h5": Path("data/cache/step_2a.h5"),
    "overwrite": True,
}


def _compute_stft(data: np.ndarray, nfft: int, hop_length: int) -> np.ndarray:
    """Apply STFT to ``(C, T)`` array, return ``(C, F, T_spec, 2)``."""
    x = torch.from_numpy(data).float()
    window = torch.hann_window(nfft)
    sxx = torch.stft(
        x, n_fft=nfft, hop_length=hop_length, window=window, return_complex=True
    )
    return torch.stack([sxx.real, sxx.imag], dim=-1).numpy()


def main(
    config_path: Path | str | None = None,
    settings: dict | None = None,
) -> None:
    if settings is None:
        settings = load_settings(config_path, default_settings)

    nfft = settings.get("nfft", default_settings["nfft"])
    hop_length = settings.get("hop_length", default_settings["hop_length"])

    input_h5 = Path(settings.get("input_h5", default_settings["input_h5"]))
    output_h5 = Path(settings.get("output_h5", default_settings["output_h5"]))
    output_h5.parent.mkdir(parents=True, exist_ok=True)

    n_samples = get_sample_count(input_h5)
    logger.info(
        f"Computing STFT (nfft={nfft}, hop={hop_length}) for {n_samples} samples"
    )

    h5 = create_step_file(
        output_h5,
        metadata={"nfft": nfft, "hop_length": hop_length, "num_samples": n_samples},
    )
    try:
        for idx, data in iter_samples(input_h5):
            sxx = _compute_stft(data, nfft, hop_length)
            write_sample(h5, idx, sxx)
    finally:
        h5.close()

    logger.info(f"Wrote {n_samples} spectrograms to {output_h5}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path)
