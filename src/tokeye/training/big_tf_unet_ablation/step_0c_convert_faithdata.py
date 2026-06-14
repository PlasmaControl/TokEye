"""Step 0c: Window raw time-series into fixed-length chunks per modality.

Reads joblib dicts from step_0b, selects channels for the configured
modality, and writes windows to an HDF5 step file + ``frame_info.csv``.
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

from .preprocess.dataset import JoblibDataset
from .utils.configuration import (
    load_input_paths,
    load_settings,
)
from .utils.hdf5_io import create_step_file, write_sample

logger = logging.getLogger(__name__)

default_settings = {
    "subseq_len": 66000,
    "input_key": "bes",
    "input_channels": [26, 28, 30, 32, 34, 36, 38, 40],
    "frame_info_path": Path("data/frame_info.csv"),
    "input_dir": Path("data/cache/step_0b_filter_faithdata"),
    "output_dir": Path("data/cache/step_0c_convert_faithdata"),
    "overwrite": True,
}


def main(
    config_path: Path | str | None = None,
    settings: dict | None = None,
) -> None:
    if settings is None:
        settings = load_settings(config_path, default_settings)

    output_dir = Path(settings.get("output_dir", default_settings["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    input_dir = Path(settings.get("input_dir", default_settings["input_dir"]))
    input_paths = load_input_paths(input_dir)
    logger.info(f"Found {len(input_paths)} input paths")

    input_key = settings.get("input_key", default_settings["input_key"])
    input_channels = settings.get("input_channels", default_settings["input_channels"])
    subseq_len = settings.get("subseq_len", default_settings["subseq_len"])

    dataset = JoblibDataset(
        file_paths=input_paths,
        input_key=[input_key, "time_ms"],
        subseq_len=subseq_len,
        validate_on_init=True,
    )
    dataset.worker_init()
    logger.info(f"Dataset length: {len(dataset)}")

    # Output HDF5
    h5_path = output_dir / "step_0c.h5"
    h5 = create_step_file(
        h5_path,
        metadata={
            "input_key": input_key,
            "num_channels": len(input_channels),
            "subseq_len": subseq_len,
            "num_samples": len(dataset),
        },
    )

    # Frame info CSV
    csv_path = Path(settings.get("frame_info_path", default_settings["frame_info_path"]))
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["shotn", "time_start", "time_end"])

            from tqdm.auto import tqdm

            for index, data in tqdm(
                enumerate(dataset), total=len(dataset), desc="Windowing Data"
            ):
                time = data[0]["time_ms"][0, 0]
                start_ms, end_ms = float(time[0]), float(time[-1])
                shotidx = dataset.subseq_index[index][0]
                shotn = Path(dataset.file_paths[shotidx]).stem.split("_")[0]
                writer.writerow([shotn, f"{start_ms:.2f}", f"{end_ms:.2f}"])

                signal = data[0][input_key][input_channels, 0]
                write_sample(h5, index, signal.numpy())
    finally:
        h5.close()

    logger.info(f"Wrote {len(dataset)} samples to {h5_path}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path)
