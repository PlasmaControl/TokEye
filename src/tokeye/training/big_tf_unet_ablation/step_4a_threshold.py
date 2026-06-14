"""Step 4a: Binary thresholding of correlation analysis output.

Key changes from original:
- ``min_size``, ``remove_bottom_rows``, ``remove_top_rows`` are auto-derived
  from spectrogram dimensions (fraction-based).
- Reads/writes HDF5 step files.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.morphology import remove_small_objects
from tqdm.auto import tqdm

from .utils.auto_params import compute_min_size, compute_row_removal
from .utils.configuration import load_settings
from .utils.hdf5_io import (
    create_step_file,
    get_sample_count,
    iter_samples,
    read_sample,
    write_sample,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

default_settings = {
    "min_size": "auto",
    "min_size_fraction": 0.0002,
    "remove_bottom_rows": "auto",
    "remove_top_rows": "auto",
    "row_removal_fraction_bottom": 0.01,
    "row_removal_fraction_top": 0.004,
    "frame_info_csv": Path("data/frame_info.csv"),
    "threshold_output_path": Path("data/thresholds.csv"),
    "input_h5": Path("data/cache/step_2b_baseline.h5"),
    "output_h5": Path("data/cache/step_4a.h5"),
    "overwrite": True,
}


def get_threshold(data: np.ndarray, adjust: float = 0.0, multiplier: int = 100) -> float:
    """Compute threshold using triangle method on cumulative distribution."""
    H, W = data.shape
    median = np.median(data)
    data_2 = data.copy()
    data_2[data < median] = median

    sorted_data = np.sort(data_2.flatten())
    data_min = sorted_data.min()
    minmax = sorted_data.max() - sorted_data.min()
    if minmax < 1e-10:
        return float(median)
    sorted_data = (sorted_data - data_min) / minmax * multiplier
    cdf_values = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    cdf_values = (cdf_values - cdf_values.min()) / (cdf_values.max() - cdf_values.min()) * multiplier * 2

    x_cdf = np.linspace(sorted_data.min(), sorted_data.max(), multiplier)
    cdf = np.interp(x_cdf, sorted_data, cdf_values)

    a = -(cdf[-1] - cdf[0]) / (x_cdf[-1] - x_cdf[0])
    b = 1.0
    c = -(cdf[0] - a * x_cdf[0])
    distances = np.abs(a * x_cdf + b * cdf + c) / np.sqrt(a**2 + b**2)
    threshold_idx = np.argmax(distances)
    binary = x_cdf[threshold_idx] / multiplier * minmax + data_min
    return float(binary + adjust * (data_2.max() - data_2.min()))


def main(
    config_path: Path | str | None = None,
    settings: dict | None = None,
) -> None:
    if settings is None:
        settings = load_settings(config_path, default_settings)

    input_h5 = Path(settings.get("input_h5", default_settings["input_h5"]))
    output_h5 = Path(settings.get("output_h5", default_settings["output_h5"]))
    output_h5.parent.mkdir(parents=True, exist_ok=True)

    csv_path = Path(settings.get("frame_info_csv", default_settings["frame_info_csv"]))
    threshold_path = Path(settings.get("threshold_output_path", default_settings["threshold_output_path"]))
    threshold_path.parent.mkdir(parents=True, exist_ok=True)

    n_samples = get_sample_count(input_h5)
    logger.info(f"Thresholding {n_samples} samples from {input_h5}")

    # Peek at first sample to get dimensions
    first_data = read_sample(input_h5, 0)
    C, H, W = first_data.shape[:3]

    # Auto parameters
    if settings.get("remove_bottom_rows", "auto") == "auto":
        remove_bottom, remove_top = compute_row_removal(
            H,
            settings.get("row_removal_fraction_bottom", 0.01),
            settings.get("row_removal_fraction_top", 0.004),
        )
    else:
        remove_bottom = int(settings["remove_bottom_rows"])
        remove_top = int(settings["remove_top_rows"])

    if settings.get("min_size", "auto") == "auto":
        min_size = compute_min_size(H, W, settings.get("min_size_fraction", 0.0002))
    else:
        min_size = int(settings["min_size"])

    # Phase 1: Compute per-shot thresholds (if frame_info available)
    threshold_dict: dict[tuple[int, int], float] = {}
    frame_info = None
    if csv_path.exists():
        frame_info = pd.read_csv(csv_path)
        shot_numbers = frame_info["shotn"].unique().tolist()

        threshold_records = []
        for shotn in tqdm(shot_numbers, desc="Computing thresholds"):
            shot_df = frame_info[frame_info["shotn"] == shotn]
            file_indices = shot_df.index.tolist()
            arrays = []
            for fi in file_indices:
                try:
                    arrays.append(read_sample(input_h5, fi))
                except Exception:
                    continue
            if not arrays:
                continue
            data = np.concatenate(arrays, axis=2)
            for ch in range(data.shape[0]):
                ch_data = data[ch]
                if ch_data.ndim == 3:
                    ch_mean = np.sqrt(np.mean(ch_data**2, axis=-1))
                else:
                    ch_mean = np.abs(ch_data)
                ch_mean[:remove_bottom] = ch_mean.mean()
                ch_mean[-remove_top:] = ch_mean.mean()
                ch_mean[:, :3] = ch_mean.mean()
                ch_mean = np.gradient(ch_mean, axis=1)
                thr = get_threshold(ch_mean, adjust=0.05)
                threshold_dict[(shotn, ch)] = thr
                threshold_records.append({"shotn": shotn, "channel": ch, "threshold": thr})

        pd.DataFrame(threshold_records).to_csv(threshold_path, index=False)

    # Phase 2: Apply thresholds
    h5_out = create_step_file(output_h5, metadata={
        "min_size": min_size,
        "remove_bottom": remove_bottom,
        "remove_top": remove_top,
        "num_samples": n_samples,
    })

    try:
        for idx, data in iter_samples(input_h5):
            C_cur = data.shape[0]
            H_cur, W_cur = data.shape[1], data.shape[2]

            shotn = None
            if frame_info is not None and idx < len(frame_info):
                shotn = int(frame_info.iloc[idx]["shotn"])

            output_mask = np.zeros((C_cur, H_cur, W_cur, 1), dtype=bool)
            for ch in range(C_cur):
                ch_data = data[ch]
                if ch_data.ndim == 3:
                    ch_mean = np.sqrt(np.mean(ch_data**2, axis=-1))
                else:
                    ch_mean = np.abs(ch_data)
                ch_mean = np.gradient(ch_mean, axis=1)

                key = (shotn, ch) if shotn is not None else None
                thr = threshold_dict.get(key) if key else None
                if thr is None:
                    thr = get_threshold(ch_mean, adjust=0.05)

                binary = ch_mean > thr
                binary[:remove_bottom] = False
                binary[-remove_top:] = False
                binary[:, -1:] = False
                binary[:, :2] = False
                binary = remove_small_objects(binary, min_size=min_size)
                output_mask[ch] = np.expand_dims(binary, axis=-1)

            write_sample(h5_out, idx, output_mask)
    finally:
        h5_out.close()

    logger.info(f"Wrote threshold masks to {output_h5}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path)
