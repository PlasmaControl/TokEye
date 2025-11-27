"""
Coherence analysis results.

NOTE: This step is currently not used in the pipeline as finding coherence can re-introduce
noise compared to the original data. In addition, data normalization needs to be kept track
of. The coherence analysis is kept for experimental purposes but is not recommended for
production use.
"""

import sys
from pathlib import Path

import joblib

import numpy as np

import logging

logger = logging.getLogger(__name__)

from .utils.configuration import (
    load_settings,
    load_input_paths,
    setup_directory,
)
from .utils.parmap import ParallelMapper

default_settings = {
    "input_dir_1": Path("data/cache/step_2b_filter_spectrogram"),
    "input_dir_2": Path("data/cache/step_3b_extract_correlation"),
    "output_dir": Path("data/cache/step_3c_coherence"),
    "overwrite": True,
}


def process_single(
    input_path: Path,
    settings: dict,
) -> None:
    data_1 = joblib.load(settings["input_dir_1"] / input_path)
    data_2 = joblib.load(settings["input_dir_2"] / input_path)

    coherence = np.abs(np.sum(data_1 * data_2.conj(), axis=-1))

    output_path = settings["output_dir"] / input_path.name
    joblib.dump(coherence, output_path)


def main(
    config_path: Path | str | None = None,
) -> None:
    settings = load_settings(config_path, default_settings)
    setup_directory(
        path=settings["output_dir"],
        overwrite=settings["overwrite"],
    )

    input_paths = load_input_paths(settings["input_dir_1"])
    mapper = ParallelMapper()

    mapper(
        process_single,
        input_paths,
        settings=settings,
    )


if __name__ == "__main__":
    # python -m autotslabel.autosegment.multichannel.step_3c_coherence
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path)
