import sys
from pathlib import Path

from tqdm.auto import tqdm

import joblib

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .utils.configuration import (
    load_settings,
    load_input_paths,
    setup_directory,
)

default_settings = {
    "reference_dir": Path("data/cache/step_2b_filter_spectrogram"),
    "input_dir": Path("data/cache/step_3a_correlation_analysis"),
    "output_dir": Path("data/cache/step_3b_extract_correlation"),
    "overwrite": True,
}


def main(config_path=None):
    settings = load_settings(config_path, default_settings)

    setup_directory(
        path=settings["output_dir"],
        overwrite=settings["overwrite"],
    )
    input_paths = load_input_paths(settings["input_dir"])

    reference_paths = load_input_paths(settings["reference_dir"])

    idx = 0
    for input_path in tqdm(input_paths, desc="Extracting Denoised Data"):
        data = joblib.load(input_path)
        B, C, H, W, Z = data.shape
        for i in range(B):
            output_path = settings["output_dir"] / reference_paths[idx].name
            data_individual = data[i]
            joblib.dump(data_individual, output_path)
            idx += 1
    print(f"Processed {len(input_paths)} files")


if __name__ == "__main__":
    # python -m autotslabel.autosegment.multichannel.step_3b_extract_correlation
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path)
