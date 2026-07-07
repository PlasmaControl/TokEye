"""Step 1: STFT spectrograms + optional model-based window filtering.

Pass 1 computes an STFT for every step_0 window (``torch.stft`` with a hann
window; real/imag stacked into float32 ``(C, F, T, 2)``). With the window
filter disabled the full set goes straight to ``out_h5`` and the frame info
is copied through unchanged. With it enabled, the full set goes to a
temporary ``*.full.h5``, pass 2 scores every window with the surrogate
U-Net, keeps the top windows per shot, and re-indexes the survivors 0..N-1
into ``out_h5`` (with a matching filtered frame info). The temporary file is
always removed.

Merged port of ``big_tf_unet_ablation`` steps 2a (STFT) and 2f (window
filtering).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import torch

from .utils.hdf5_io import (
    create_step_file,
    get_sample_count,
    iter_samples,
    read_sample,
    write_sample,
)
from .utils.window_filter import (
    compute_logmag_stats,
    load_filter_model,
    score_window,
    select_window_indices,
)

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


def _compute_stft(data: np.ndarray, nfft: int, hop: int) -> np.ndarray:
    """STFT of a ``(C, T)`` window -> float32 ``(C, F, T_spec, 2)`` (real/imag)."""
    x = torch.from_numpy(data).float()
    window = torch.hann_window(nfft)
    sxx = torch.stft(
        x, n_fft=nfft, hop_length=hop, window=window, return_complex=True
    )
    return torch.stack([sxx.real, sxx.imag], dim=-1).numpy()


def _write_spectrograms(in_h5: Path, out_h5: Path, nfft: int, hop: int) -> int:
    """Pass 1: STFT every sample of ``in_h5`` into a new step file at ``out_h5``."""
    n = get_sample_count(in_h5)
    logger.info(f"computing STFT (nfft={nfft}, hop={hop}) for {n} windows")
    h5 = create_step_file(
        out_h5, metadata={"nfft": nfft, "hop": hop, "num_samples": n}
    )
    try:
        for idx, data in iter_samples(in_h5):
            write_sample(h5, idx, _compute_stft(data, nfft, hop))
    finally:
        h5.close()
    return n


def _filter_windows(
    full_h5: Path,
    out_h5: Path,
    frame_info_in: Path,
    frame_info_out: Path,
    nfft: int,
    hop: int,
    settings: dict,
) -> dict | None:
    """Pass 2: score every window, keep the most active per shot, re-index."""
    device = "cpu"  # inline step (login node): never grab a GPU
    fi = pd.read_csv(frame_info_in)  # row order aligns with sample idx

    # Per-modality normalization: each diagnostic has its own intensity scale
    # and 1/f structure, so "auto" computes this modality's own log-magnitude
    # stats from the freshly computed spectrograms.
    mean, std = settings["mean"], settings["std"]
    stats_computed = mean == "auto" or std == "auto"
    if stats_computed:
        mean, std = compute_logmag_stats(full_h5)
    mean, std = float(mean), float(std)
    logger.info(
        f"window-filter normalization (per modality): mean={mean:.4f} std={std:.4f}"
    )

    model = load_filter_model(settings["weights"], device=device)
    threshold = float(settings["activity_threshold"])
    n = get_sample_count(full_h5)
    with torch.no_grad():
        scores = {
            idx: score_window(
                model, read_sample(full_h5, idx), mean, std, threshold, device
            )
            for idx in range(n)
        }

    max_windows = int(settings["max_windows_per_shot"])
    min_activity = float(settings["min_activity"])
    kept: list[int] = []
    for shotn, grp in fi.groupby("shotn"):
        shot_scores = {i: scores[i] for i in grp.index if i in scores}
        selected = select_window_indices(shot_scores, max_windows, min_activity)
        kept.extend(selected)
        logger.info(f"shot {shotn}: kept {len(selected)}/{len(grp)} windows")
    kept.sort()
    logger.info(f"window filter: kept {len(kept)}/{n} windows total")

    h5 = create_step_file(
        out_h5,
        metadata={"nfft": nfft, "hop": hop, "filtered": True, "kept": len(kept)},
    )
    try:
        for new_idx, old_idx in enumerate(kept):
            write_sample(h5, new_idx, read_sample(full_h5, old_idx))
    finally:
        h5.close()

    # Filtered frame info: new contiguous index, original sample idx kept as
    # orig_index, all original columns (shotn, window_start, ...) carried over.
    filtered = fi.iloc[kept].reset_index(drop=True)
    filtered["orig_index"] = kept
    filtered["index"] = filtered.index
    lead = ["index", "orig_index"]
    cols = lead + [c for c in filtered.columns if c not in lead]
    filtered[cols].to_csv(frame_info_out, index=False)
    logger.info(f"wrote {out_h5} and {frame_info_out}")

    if stats_computed:
        return {"mean": mean, "std": std}
    return None


def main(settings: dict) -> dict | None:
    in_h5 = Path(settings["in_h5"])
    out_h5 = Path(settings["out_h5"])
    frame_info_in = Path(settings["frame_info_in"])
    frame_info_out = Path(settings["frame_info_out"])
    nfft = int(settings["nfft"])
    hop = int(settings["hop"])

    if not settings["filter_enabled"]:
        _write_spectrograms(in_h5, out_h5, nfft, hop)
        shutil.copyfile(frame_info_in, frame_info_out)
        logger.info("window filter disabled: wrote unfiltered spectrograms")
        return None

    full_h5 = out_h5.with_suffix(".full.h5")
    try:
        _write_spectrograms(in_h5, full_h5, nfft, hop)
        return _filter_windows(
            full_h5, out_h5, frame_info_in, frame_info_out, nfft, hop, settings
        )
    finally:
        # Not a registered artifact -- it must never survive the step.
        full_h5.unlink(missing_ok=True)
