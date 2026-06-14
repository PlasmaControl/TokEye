"""Step 6a: convert per-window HDF5 outputs to dual-mask TIF pairs.

For each modality, reads three index-aligned HDF5 files:
  - img  : shared step_2a_filtered.h5   (C, F, T, 2)  complex spectrogram
  - coh  : step_4a_threshold.h5         (C, H, W, 1)  coherent mask  (ch0 target)
  - tra  : step_4a_threshold_baseline.h5(C, H, W, 1)  transient mask (ch1 target)

Per-modality magnitude statistics standardize the image; each (window, channel)
pair becomes one ``{idx}_img.tif`` (1, H, W) + ``{idx}_mask.tif`` (2, H, W) with
mask channel 0 = coherent, channel 1 = transient. This reproduces the canonical
2-channel target of the original ``big_tf_unet`` pipeline (verified against
``step_6a.validate_directory``) but over the ablation's HDF5 layout.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import tifffile as tif

from .utils.configuration import load_settings
from .utils.hdf5_io import get_sample_count, read_sample

logger = logging.getLogger(__name__)

default_settings = {
    "modality_inputs": [],  # list of {name, img_h5, coh_h5, tra_h5}
    "output_dir": Path("data/cache/ablation/full/step_6a"),
    "zscore_clip": 3,
    "overwrite": True,
}


def process_data_img(
    data: np.ndarray, channel_idx: int, stats: dict, zscore_clip: float
) -> np.ndarray:
    """Magnitude, log1p, clip to mean +/- zscore*std, standardize. data: (F, T, 2)."""
    magnitude = np.sqrt(data[..., 0] ** 2 + data[..., 1] ** 2).astype(np.float32)
    magnitude = np.log1p(magnitude)
    mean = stats["means"][channel_idx]
    std = stats["stds"][channel_idx]
    lo, hi = mean - zscore_clip * std, mean + zscore_clip * std
    magnitude = np.clip(magnitude, lo, hi)
    magnitude = (magnitude - mean) / (std + 1e-8)
    return magnitude.astype(np.float32)


def process_data_mask_dual(
    data_coherent: np.ndarray, data_transient: np.ndarray
) -> np.ndarray:
    """Stack coherent + transient masks as (2, H, W). Inputs: (H, W, 1)."""
    mask_coh = data_coherent[..., 0].astype(np.float32)
    mask_tra = data_transient[..., 0].astype(np.float32)
    mask_coh[-4:] = 0  # fixed edge guard (from original step_6a)
    return np.stack([mask_coh, mask_tra], axis=0)


def _collect_stats(img_h5: Path) -> dict:
    """Per-channel mean/std of log-magnitude across all windows in one modality."""
    n = get_sample_count(img_h5)
    first = read_sample(img_h5, 0)
    C = first.shape[0]
    sums = np.zeros(C)
    sq = np.zeros(C)
    cnt = np.zeros(C)
    for idx in range(n):
        data = read_sample(img_h5, idx)
        for c in range(C):
            mag = np.log1p(np.sqrt(data[c, ..., 0] ** 2 + data[c, ..., 1] ** 2))
            sums[c] += mag.sum()
            sq[c] += (mag**2).sum()
            cnt[c] += mag.size
    means = sums / cnt
    stds = np.sqrt(np.maximum(sq / cnt - means**2, 1e-12))
    return {"means": means.tolist(), "stds": stds.tolist()}


def main(config_path=None, settings=None):
    if settings is None:
        settings = load_settings(config_path, default_settings)

    output_dir = Path(settings["output_dir"])
    if output_dir.exists() and settings.get("overwrite", True):
        import shutil

        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    modality_inputs = settings["modality_inputs"]
    stats_file = output_dir.parent / "normalization_statistics.txt"
    global_idx = 0

    with stats_file.open("w") as sf:
        sf.write("# Per-modality, per-channel log-magnitude normalization stats\n")

    for mod in modality_inputs:
        img_h5 = Path(mod["img_h5"])
        coh_h5 = Path(mod["coh_h5"])
        tra_h5 = Path(mod["tra_h5"])
        name = mod.get("name", img_h5.parent.name)
        if not (img_h5.exists() and coh_h5.exists() and tra_h5.exists()):
            logger.warning(
                f"skip modality {name}: missing one of {img_h5} / {coh_h5} / {tra_h5}"
            )
            continue

        stats = _collect_stats(img_h5)
        with stats_file.open("a") as sf:
            sf.write(f"Modality: {name}\n")
            for c, (m, s) in enumerate(zip(stats["means"], stats["stds"], strict=False)):
                sf.write(f"  Channel {c}: mean={m:.6f}, std={s:.6f}\n")

        n = get_sample_count(img_h5)
        logger.info(f"step_6a [{name}]: {n} windows")
        for idx in range(n):
            img = read_sample(img_h5, idx)  # (C, F, T, 2)
            coh = read_sample(coh_h5, idx)  # (C, H, W, 1)
            tra = read_sample(tra_h5, idx)  # (C, H, W, 1)
            C = min(img.shape[0], coh.shape[0], tra.shape[0])
            for c in range(C):
                img_out = process_data_img(img[c], c, stats, settings.get("zscore_clip", 3))
                mask_out = process_data_mask_dual(coh[c], tra[c])
                tif.imwrite(
                    str(output_dir / f"{global_idx}_img.tif"),
                    img_out[np.newaxis, ...],
                    imagej=True,
                )
                tif.imwrite(
                    str(output_dir / f"{global_idx}_mask.tif"),
                    mask_out,
                    imagej=True,
                )
                global_idx += 1

    logger.info(f"step_6a: wrote {global_idx} TIF pairs to {output_dir}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
