"""Step 2f: filter windows by model activity -> reduced step_2a_filtered.h5.

Invoked once per modality. Groups windows by shot via frame_info.csv (written by
step_0c), scores each window with the existing TokEye model, keeps
<= max_windows_per_shot above the activity floor, and writes a new HDF5 with only
kept windows (re-indexed 0..N-1) plus a filtered frame_info.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

from .utils.configuration import load_settings
from .utils.hdf5_io import (
    create_step_file,
    get_sample_count,
    iter_samples,
    read_sample,
    write_sample,
)
from .window_filter import (
    compute_logmag_stats,
    load_filter_model,
    score_window,
    select_window_indices,
)

logger = logging.getLogger(__name__)

default_settings = {
    "input_h5": Path("data/cache/ablation/shared/bes/step_2a.h5"),
    "output_h5": Path("data/cache/ablation/shared/bes/step_2a_filtered.h5"),
    "frame_info_csv": Path("data/cache/ablation/shared/bes/frame_info.csv"),
    "frame_info_out": Path("data/cache/ablation/shared/bes/frame_info_filtered.csv"),
    "enabled": True,
    "max_windows_per_shot": 25,
    "activity_threshold": 0.5,
    "min_activity": 0.0005,
    "mean": 17.84620821169868,
    "std": 25.016818830630463,
    "weights": "/scratch/gpfs/nc1514/aemodes/model/big_mode_v1-5_weights.pt",
    "weights_fallback": "model/big_tf_unet_251210_weights.pt",
}


def main(config_path=None, settings=None):
    import torch

    if settings is None:
        settings = load_settings(config_path, default_settings)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_h5 = Path(settings["input_h5"])
    out_h5 = Path(settings["output_h5"])
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    fi = pd.read_csv(settings["frame_info_csv"])  # row index aligns with sample idx

    if not settings.get("enabled", True):
        h5 = create_step_file(out_h5, metadata={"filtered": False})
        try:
            for idx, data in iter_samples(in_h5):
                write_sample(h5, idx, data)
        finally:
            h5.close()
        fi.to_csv(settings["frame_info_out"], index=False)
        logger.info("window filter disabled: passthrough copy")
        return

    model = load_filter_model(settings["weights"], settings.get("weights_fallback"), device)
    n = get_sample_count(in_h5)

    # Per-modality normalization: each diagnostic has its own intensity scale /
    # 1/f structure, so compute this modality's log-magnitude mean/std (auto)
    # unless explicitly overridden. Using a single global value mis-normalizes
    # the non-reference diagnostics and corrupts their activity scores.
    mean, std = settings.get("mean", "auto"), settings.get("std", "auto")
    if mean == "auto" or std == "auto":
        mean, std = compute_logmag_stats(in_h5)
    logger.info(f"window-filter normalization (per modality): mean={mean:.4f} std={std:.4f}")

    scores: dict[int, float] = {}
    for idx in range(n):
        scores[idx] = score_window(
            model,
            read_sample(in_h5, idx),
            mean,
            std,
            settings["activity_threshold"],
            device,
        )

    kept: list[int] = []
    for _shotn, grp in fi.groupby("shotn"):
        shot_scores = {i: scores[i] for i in grp.index if i in scores}
        kept.extend(
            select_window_indices(
                shot_scores,
                int(settings["max_windows_per_shot"]),
                float(settings["min_activity"]),
            )
        )
    kept.sort()
    logger.info(f"window filter: kept {len(kept)}/{n} windows")

    h5 = create_step_file(out_h5, metadata={"filtered": True, "kept": len(kept)})
    try:
        for new_idx, old_idx in enumerate(kept):
            write_sample(h5, new_idx, read_sample(in_h5, old_idx))
    finally:
        h5.close()
    fi.iloc[kept].reset_index(drop=True).to_csv(settings["frame_info_out"], index=False)
    logger.info(f"wrote {out_h5}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
