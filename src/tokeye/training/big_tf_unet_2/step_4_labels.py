"""step_4 — label masks from knee-point thresholds on robust-z fields.

Two label channels per window, thresholded per (shot, channel) so a strong
shot cannot set another shot's threshold:

- **coherent** (channel 0): the denoised field from step_3,
- **transient** (channel 1): |dt B| of the step_2 baseline — broadband bursts
  move the baseline level between neighboring time columns.

Both fields are robust-standardized ((x - median) / (1.4826*MAD), pooled over
the shot's windows per channel) and thresholded at the Kneedle knee of the
positive-z ECDF (``kneed``), replacing the hand-rolled triangle method. The
threshold lives in robust-sigma units, so knobs are comparable across shots,
channels, modalities, and scales. No asinh here: knee location is not
invariant under monotone compression.

Output ``/samples/{idx}``: uint8 ``(C, 2, F, T)`` (coherent, transient).
Every (shot, channel, target) threshold is logged to ``thresholds.csv``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import h5py
import numpy as np
import pandas as pd
from skimage.morphology import remove_small_objects

from .utils.auto_params import knee_threshold, robust_stats
from .utils.hdf5_io import create_step_file, read_sample, write_sample

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)


def _magnitude(arr: np.ndarray) -> np.ndarray:
    """(C, F, T, 2) real/imag -> (C, F, T) magnitude; (C, F, T) passes through."""
    if arr.ndim == 4 and arr.shape[-1] == 2:
        return np.sqrt(arr[..., 0] ** 2 + arr[..., 1] ** 2)
    return np.abs(arr)


def _transient_field(baseline: np.ndarray) -> np.ndarray:
    """|dt| of the per-channel mean baseline level, width-preserving."""
    level = baseline.mean(axis=-1) if baseline.ndim == 4 else baseline  # (C, F, T)
    return np.abs(np.diff(level, axis=-1, prepend=level[..., :1]))


def _postprocess(mask: np.ndarray, settings: dict) -> np.ndarray:
    """Row removal + small-object removal on one (F, T) bool mask."""
    bottom = settings["remove_bottom_rows"]
    top = settings["remove_top_rows"]
    if bottom > 0:
        mask[:bottom] = False
    if top > 0:
        mask[-top:] = False
    return remove_small_objects(mask, min_size=settings["min_size"])


def _label_shot(
    fields: np.ndarray,  # (W, C, F, T) one target field for one shot
    settings: dict,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Threshold one target for one shot. Returns (W, C, F, T) bool + rows."""
    n_windows, n_channels = fields.shape[:2]
    masks = np.zeros(fields.shape, dtype=bool)
    rows = []
    for c in range(n_channels):
        pooled = fields[:, c]  # (W, F, T)
        med, scale = robust_stats(pooled)
        z = (pooled - med) / scale
        result = knee_threshold(
            z,
            sensitivity=settings["knee_sensitivity"],
            delta=settings["delta"],
            fallback_frac=settings["fallback_frac"],
        )
        raw = z > result["threshold"]
        for w in range(n_windows):
            masks[w, c] = _postprocess(raw[w], settings)
        rows.append({"channel": c, **result})
    return masks, rows


def main(settings: dict) -> None:
    frame_info = pd.read_csv(settings["frame_info"])
    by_shot = frame_info.groupby("shotn")["index"].apply(list)

    threshold_rows: list[dict[str, Any]] = []
    out = create_step_file(settings["out_h5"], metadata={"targets": "coherent,transient"})
    try:
        with (
            h5py.File(settings["denoised_h5"], "r") as f_den,
            h5py.File(settings["baseline_h5"], "r") as f_base,
        ):
            for shot, indices in by_shot.items():
                coherent = np.stack(
                    [_magnitude(read_sample(f_den, i)) for i in indices]
                )
                transient = np.stack(
                    [_transient_field(read_sample(f_base, i)) for i in indices]
                )
                shot_masks = []
                for target, fields in (
                    ("coherent", coherent),
                    ("transient", transient),
                ):
                    masks, rows = _label_shot(fields, settings)
                    shot_masks.append(masks)
                    for row in rows:
                        threshold_rows.append({"shotn": shot, "target": target, **row})
                # (W, C, F, T) x2 -> per window (C, 2, F, T)
                stacked = np.stack(shot_masks, axis=2).astype(np.uint8)
                for w, idx in enumerate(indices):
                    write_sample(out, idx, stacked[w])
                logger.info(
                    f"shot {shot}: {len(indices)} windows labeled "
                    f"(coherent+transient, {coherent.shape[1]} channels)"
                )
    finally:
        out.close()

    pd.DataFrame(threshold_rows).to_csv(settings["thresholds_csv"], index=False)
    logger.info(f"thresholds -> {settings['thresholds_csv']}")
