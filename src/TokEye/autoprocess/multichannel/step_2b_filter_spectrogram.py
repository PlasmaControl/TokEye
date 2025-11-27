import sys
from pathlib import Path

import joblib

import numpy as np

from pybaselines import Baseline2D

import logging

logger = logging.getLogger(__name__)

from .utils.configuration import (
    load_settings,
    load_input_paths,
    setup_directory,
)
from .utils.parmap import ParallelMapper

default_settings = {
    "baseline_method": "fabc",
    "baseline_method_kwargs": {"lam": 1e5},
    "input_dir": Path("data/cache/step_2a_make_spectrogram"),
    "output_dir": Path("data/cache/step_2b_filter_spectrogram"),
    "output_baseline_dir": Path("data/cache/step_2b_filter_spectrogram_baseline"),
    "overwrite": True,
}


def filter_baseline(
    data,
    method="arpls",
    method_kwargs={"lam": 1e6},
):
    H, W = data.shape

    x = np.arange(data.shape[0])
    y = np.arange(data.shape[1])

    baseline_fitter = Baseline2D(x, y)
    baseline, params = baseline_fitter.individual_axes(
        data,
        axes=0,
        method=method,
        method_kwargs=method_kwargs,
    )

    return baseline


def process_single_rotation(
    data: np.ndarray,
    settings: dict,
) -> np.ndarray:
    H, W = data.shape
    sxx = data
    sxx = np.abs(sxx)
    sxx = np.log1p(sxx)

    lower_idx = 3
    upper_idx = 2
    lower_mean = sxx[:, lower_idx].mean()
    upper_mean = sxx[:, -upper_idx].mean()
    sxx[:lower_idx] = lower_mean
    sxx[-upper_idx + 1 :] = upper_mean

    baseline = filter_baseline(
        sxx,
        settings["baseline_method"],
        settings["baseline_method_kwargs"],
    )

    sxx = (sxx - baseline) / (baseline + 1e-6)

    return sxx, baseline


def process_single_channel(
    data: np.ndarray,
    settings: dict,
) -> np.ndarray:
    H, W, Z = data.shape

    sxx_real = data[..., 0]
    sxx_imag = data[..., 1]

    sxx_real, baseline_real = process_single_rotation(sxx_real, settings)
    sxx_imag, baseline_imag = process_single_rotation(sxx_imag, settings)

    data_out = np.stack([sxx_real, sxx_imag], axis=-1)
    return data_out, np.stack([baseline_real, baseline_imag], axis=-1)


def process_single(
    input_path: Path,
    settings: dict,
) -> None:
    data = joblib.load(input_path)
    C, H, W, Z = data.shape
    output_data = np.zeros((C, H, W, Z))
    output_baseline = np.zeros((C, H, W, Z))
    output_path = settings["output_dir"] / input_path.name
    output_baseline_path = settings["output_baseline_dir"] / input_path.name
    try:
        for i in range(C):
            output_data[i], output_baseline[i] = process_single_channel(
                data[i],
                settings,
            )
        joblib.dump(output_data, output_path)
        joblib.dump(
            output_baseline,
            output_baseline_path,
        )
    except Exception as e:
        joblib.dump(output_data, output_path)
        joblib.dump(output_baseline, output_baseline_path)
        logger.error(f"Error processing file {input_path}: {e}")


def main(
    config_path: Path | str | None = None,
) -> None:
    settings = load_settings(config_path, default_settings)
    setup_directory(
        path=settings["output_dir"],
        overwrite=settings["overwrite"],
    )
    setup_directory(
        path=settings["output_baseline_dir"],
        overwrite=settings["overwrite"],
    )

    input_paths = load_input_paths(settings["input_dir"])
    mapper = ParallelMapper()

    mapper(
        process_single,
        input_paths,
        settings=settings,
    )


if __name__ == "__main__":
    # python -m autotslabel.autosegment.multichannel.step_2b_filter_spectrogram
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path)
