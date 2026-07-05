"""Headless batch inference: run TokEye over a list of files with no GUI.

Never imports gradio, so this module (and anything that imports only this
module) is safe to run on HPC login/compute nodes and in CI. ``matplotlib``
is switched to the non-interactive ``Agg`` backend before ``pyplot`` is
imported, so this stays headless-safe even without a display.
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path

import matplotlib as mpl
import numpy as np
from tqdm.auto import tqdm

from . import hub
from .inference import model_infer, signal_to_spectrogram
from .transforms import (
    DEFAULT_CLIP_DC,
    DEFAULT_CLIP_HIGH,
    DEFAULT_CLIP_LOW,
    DEFAULT_HOP,
    DEFAULT_N_FFT,
)

mpl.use("Agg")  # Must precede the pyplot import below (headless HPC safety).

import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)


def collect_inputs(inputs: list[str]) -> list[Path]:
    """Expand a list of files/directories/glob patterns into concrete paths.

    Each item is resolved as: an existing file (kept as-is), an existing
    directory (its ``*.npy`` files, sorted), or otherwise a glob pattern
    (matches, sorted). Duplicates are dropped, preserving first-seen order.
    """
    collected: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_file():
            found = [path]
        elif path.is_dir():
            found = sorted(path.glob("*.npy"))
        else:
            # glob.glob (not Path.glob) so absolute patterns keep working:
            # Path(".").glob() rejects non-relative patterns outright.
            found = sorted(Path(match) for match in glob.glob(item))  # noqa: PTH207
        collected.extend(found)

    seen: set[Path] = set()
    result: list[Path] = []
    for path in collected:
        if path not in seen:
            seen.add(path)
            result.append(path)

    if not result:
        raise ValueError(f"No input files found for: {inputs}")

    return result


def load_input(path: Path, stft_kwargs: dict) -> np.ndarray:
    """Load a ``.npy`` file as a spectrogram, computing one if it's a signal."""
    arr = np.load(path)
    if arr.ndim == 1:
        return signal_to_spectrogram(arr, **stft_kwargs)
    if arr.ndim == 2:
        return arr.astype(float)
    raise ValueError(f"expected 1D signal or 2D spectrogram, got ndim={arr.ndim}")


def save_overlay_png(
    spectrogram: np.ndarray,
    mask: np.ndarray,
    out_path: Path,
    threshold: float = 0.5,
    dpi: int = 150,
) -> None:
    """Save a grayscale spectrogram with a semi-transparent mask overlay.

    Coherent activity (``mask[0]``) is tinted green, transient activity
    (``mask[1]``) is tinted red, both thresholded at ``threshold``.
    """
    height, width = spectrogram.shape
    overlay = np.zeros((height, width, 4), dtype=np.float32)
    overlay[mask[0] >= threshold] = (0.0, 1.0, 0.0, 0.4)
    overlay[mask[1] >= threshold] = (1.0, 0.0, 0.0, 0.4)

    fig, ax = plt.subplots()
    ax.imshow(spectrogram, cmap="gray", origin="lower", aspect="auto")
    ax.imshow(overlay, origin="lower", aspect="auto")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def process_file(
    path: Path,
    model,
    stft_kwargs: dict,
    out_dir: Path,
    save_png: bool = True,
    threshold: float = 0.5,
) -> Path:
    """Run inference on a single input file and write its outputs to disk."""
    spectrogram = load_input(path, stft_kwargs)
    mask = model_infer(spectrogram, model)

    mask_path = out_dir / f"{path.stem}_mask.npy"
    np.save(mask_path, mask.astype(np.float32))

    if save_png:
        preview_path = out_dir / f"{path.stem}_preview.png"
        save_overlay_png(spectrogram, mask, preview_path, threshold=threshold)

    return mask_path


def run_batch(
    inputs: list[str],
    model: str | Path = hub.DEFAULT_MODEL,
    out_dir: Path = Path("tokeye_output"),
    stft_kwargs: dict | None = None,
    save_png: bool = True,
    threshold: float = 0.5,
    device: str = "auto",
) -> int:
    """Run inference over ``inputs``, writing masks (and previews) to ``out_dir``.

    Returns the number of files that failed to process.
    """
    paths = collect_inputs(inputs)
    resolved_stft_kwargs = stft_kwargs if stft_kwargs is not None else {
        "n_fft": DEFAULT_N_FFT,
        "hop": DEFAULT_HOP,
        "clip_dc": DEFAULT_CLIP_DC,
        "clip_low": DEFAULT_CLIP_LOW,
        "clip_high": DEFAULT_CLIP_HIGH,
    }

    loaded_model = hub.load_model(model, device)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for path in tqdm(paths, desc="tokeye run"):
        try:
            process_file(
                path,
                loaded_model,
                resolved_stft_kwargs,
                out_dir,
                save_png=save_png,
                threshold=threshold,
            )
        except Exception:
            logger.error("Failed to process %s", path, exc_info=True)
            failures += 1

    return failures
