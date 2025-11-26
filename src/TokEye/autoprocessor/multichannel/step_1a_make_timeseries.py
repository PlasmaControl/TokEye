import sys
from pathlib import Path

import joblib

import logging
logger = logging.getLogger(__name__)

from .utils.configuration import (
    load_settings, 
    load_input_paths, 
    setup_directory,
    )
from .utils.parmap import ParallelMapper

default_settings = {
    'input_dir': Path('data/cache/step_0c_convert_faithdata'),
    'output_dir': Path('data/cache/step_1a_make_timeseries'),
    'overwrite': True,
    'make_original': True,
}

def _setup_original(
    settings: dict,
) -> dict:
    """Make original data."""
    if settings['make_original']:
        input_dir, output_dir = settings['input_dir'], settings['output_dir']
        settings['input_dir'] = input_dir.parent / 'original' / input_dir.name
        settings['output_dir'] = output_dir.parent / 'original' / output_dir.name
    return settings

def process_single(
    input_path: Path,
    output_dir: Path,
    settings: dict,
    ) -> None:
    """Process a single input path."""

    data = joblib.load(input_path)
    C, T = data.shape

    output_path = output_dir / input_path.name
    joblib.dump(data, output_path, compress=True)

def main(
    config_path: Path | str | None = None,
    ) -> None:
    settings = load_settings(config_path, default_settings)
    settings = _setup_original(settings)
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
    # python -m autotslabel.autosegment.multichannel.step_1a_make_timeseries
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path)