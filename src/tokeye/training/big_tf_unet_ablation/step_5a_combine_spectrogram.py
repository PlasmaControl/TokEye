"""Step 5a: Combine time-windowed samples into full-shot files.

Reads per-window spectrograms (step_2a) and masks (step_4a) from HDF5,
concatenates along the time axis per shot per channel, and writes
combined shot data to a step_5a HDF5.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .utils.configuration import load_settings
from .utils.hdf5_io import (
    create_step_file,
    read_sample,
    write_sample,
)

logger = logging.getLogger(__name__)

default_settings = {
    "frame_info_csv": Path("data/frame_info.csv"),
    "input_img_h5": Path("data/cache/step_2a.h5"),
    "input_mask_h5": Path("data/cache/step_4a.h5"),
    "output_h5": Path("data/cache/step_5a.h5"),
    "overwrite": True,
}


def main(
    config_path: Path | str | None = None,
    settings: dict | None = None,
) -> None:
    if settings is None:
        settings = load_settings(config_path, default_settings)

    csv_path = Path(settings.get("frame_info_csv", default_settings["frame_info_csv"]))
    input_img_h5 = Path(settings.get("input_img_h5", default_settings["input_img_h5"]))
    input_mask_h5 = Path(settings.get("input_mask_h5", default_settings["input_mask_h5"]))
    output_h5 = Path(settings.get("output_h5", default_settings["output_h5"]))
    output_h5.parent.mkdir(parents=True, exist_ok=True)

    frame_info = pd.read_csv(csv_path)
    shot_numbers = frame_info["shotn"].unique().tolist()
    logger.info(f"Combining {len(shot_numbers)} shots")

    h5_out = create_step_file(output_h5)
    sample_idx = 0

    try:
        for shotn in tqdm(shot_numbers, desc="Combining shots"):
            shot_df = frame_info[frame_info["shotn"] == shotn]
            file_indices = shot_df.index.tolist()
            if not file_indices:
                continue

            # Load images
            img_list = []
            for fi in file_indices:
                try:
                    img_list.append(read_sample(input_img_h5, fi))
                except Exception:
                    continue
            if not img_list:
                continue

            # Load masks
            mask_list = []
            for fi in file_indices:
                try:
                    mask_list.append(read_sample(input_mask_h5, fi))
                except Exception:
                    continue

            num_channels = img_list[0].shape[0]

            for ch in range(num_channels):
                # Concatenate images along time (axis=2 for (C,F,T,2) → single channel (F,T,2))
                ch_imgs = [img[ch] for img in img_list]
                combined_img = np.concatenate(ch_imgs, axis=1)  # (F, T_total, 2)

                # Convert to magnitude
                complex_data = combined_img[..., 0] + 1j * combined_img[..., 1]
                magnitude = np.log1p(np.abs(complex_data))  # (F, T_total)

                # Write as "{shotn}_{ch}"
                write_sample(h5_out, sample_idx, magnitude, group="images")

                # Concatenate masks
                if mask_list:
                    ch_masks = [m[ch] for m in mask_list]
                    combined_mask = np.concatenate(ch_masks, axis=1)  # (H, T_total, 1)
                    combined_mask = combined_mask.squeeze(axis=-1)  # (H, T_total)
                    write_sample(h5_out, sample_idx, combined_mask, group="masks")

                # Store shot/channel info
                h5_out.require_group("metadata_per_sample")
                meta = h5_out["metadata_per_sample"]
                meta.create_dataset(str(sample_idx), data=np.array([shotn, ch]))
                sample_idx += 1
    finally:
        h5_out.attrs["num_samples"] = sample_idx
        h5_out.close()

    logger.info(f"Wrote {sample_idx} combined shot-channel pairs to {output_h5}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path)
