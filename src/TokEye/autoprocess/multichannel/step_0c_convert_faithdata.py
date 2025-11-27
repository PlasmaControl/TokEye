import sys
from pathlib import Path

from tqdm.auto import tqdm

import joblib
import csv

import torch

import logging

logger = logging.getLogger(__name__)

from faith.train.data.datasets.file_based import JoblibDataset

from .utils.configuration import (
    load_settings,
    load_input_paths,
    setup_directory,
)

default_settings = {
    "subseq_len": 66000,
    "input_key": "bes",
    "input_channels": [26, 28, 30, 32, 34, 36, 38, 40],
    "frame_info_path": Path("data/frame_info.csv"),
    "input_dir": Path("data/cache/step_0b_filter_faithdata"),
    "output_dir": Path("data/cache/step_0c_convert_faithdata"),
    "overwrite": True,
    "make_original": False,
}


def _setup_original(
    settings: dict,
) -> dict:
    """Make original data."""
    if settings["make_original"]:
        input_dir, output_dir = settings["input_dir"], settings["output_dir"]
        settings["input_dir"] = input_dir.parent / "step_0a_extract_faithdata"
        settings["output_dir"] = output_dir.parent / "original" / output_dir.name
    return settings


def process_single(
    signal: torch.Tensor,
    output_name: str,
    settings: dict,
) -> None:
    """Process a single signal and save it to a joblib file."""
    C, T = signal.shape
    signal = signal.numpy()

    output_path = settings["output_dir"] / f"{output_name}.joblib"
    joblib.dump(signal, output_path, compress=True)


def main(config_path=None):
    settings = load_settings(config_path, default_settings)
    settings = _setup_original(settings)
    setup_directory(
        path=settings["output_dir"],
        overwrite=settings["overwrite"],
    )

    input_paths = load_input_paths(settings["input_dir"])
    logger.info(f"Found {len(input_paths)} input paths")

    dataset = JoblibDataset(
        file_paths=input_paths,
        input_key=[settings["input_key"], "time_ms"],
        subseq_len=settings["subseq_len"],
        validate_on_init=True,
    )
    dataset.worker_init()
    logger.info(f"Length of dataset: {len(dataset)}")

    csv_path = settings["frame_info_path"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["shotn", "time_start", "time_end"])

        for index, data in tqdm(
            enumerate(dataset),
            total=len(dataset),
            desc="Windowing Data",
        ):
            time = data[0]["time_ms"][0, 0]
            start_ms, end_ms = time[[0, -1]]
            shotidx = dataset.subseq_index[index][0]
            shotn = Path(dataset.file_paths[shotidx]).stem.split("_")[0]
            writer.writerow([shotn, f"{start_ms:.2f}", f"{end_ms:.2f}"])

            signal = data[0][settings["input_key"]][settings["input_channels"], 0]
            process_single(
                signal=signal,
                output_name=f"{index}",
                settings=settings,
            )

    logger.info("Done")


if __name__ == "__main__":
    # python -m autotslabel.autosegment.multichannel.step_0c_convert_faithdata
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path)
