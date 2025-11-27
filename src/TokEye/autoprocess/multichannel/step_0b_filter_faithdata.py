import sys
from pathlib import Path

import joblib

import torch
from torchaudio.transforms import Preemphasis

import logging
logger = logging.getLogger(__name__)

from .utils.configuration import (
    load_settings, 
    setup_directory, 
    load_input_paths,
    )
from .utils.parmap import ParallelMapper

default_settings = {
    'preemphasis_coeff': 0.99,
    'input_dir': Path('data/cache/step_0a_extract_faithdata'),
    'output_dir': Path('data/cache/step_0b_filter_faithdata'),
    'overwrite': False,
}

def process_single(
    input_path: Path,
    output_dir: Path,
    settings: dict,
    ) -> None:
    """Process a single input path."""

    data = joblib.load(input_path)
    key = [k for k in data.keys() if k != 'time_ms'][0]
    
    filtered_data = data[key]
    filtered_data = torch.from_numpy(filtered_data)

    preemphasis = Preemphasis(settings['preemphasis_coeff'])
    filtered_data = preemphasis(filtered_data)
    filtered_data = filtered_data.numpy()

    data[key] = filtered_data

    output_path = output_dir / input_path.name
    joblib.dump(data, output_path, compress=False)


def main(
    config_path: Path | str | None = None,
    ) -> None:
    settings = load_settings(config_path, default_settings)
    setup_directory(
        path=settings['output_dir'], 
        overwrite=settings['overwrite'],
    )
    input_paths = load_input_paths(settings['input_dir'])
    mapper = ParallelMapper()

    mapper(
        process_single,
        input_paths, 
        output_dir=settings['output_dir'], 
        settings=settings,
    )

if __name__ == '__main__':
    # python -m autotslabel.autosegment.multichannel.step_0b_filter_faithdata
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path)