import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import label

import logging
logger = logging.getLogger(__name__)

from .configuration import load_settings, setup_directory
from .parmap import ParallelMapper

# Step-to-directory mappings
STEP_MAPPINGS = {
    'original': {
        'input': Path('data/cache/original/step_2a_make_spectrogram'),
        'output': 'original'
    },
    'filtered': {
        'input': Path('data/cache/step_2a_make_spectrogram'),
        'output': 'enhanced/filtered'
    },
    'whitened': {
        'input': Path('data/cache/step_2b_filter_spectrogram'),
        'output': 'enhanced/whitened'
    },
    'denoised': {
        'input': Path('data/cache/step_3b_extract_correlation'),
        'output': 'enhanced/denoised'
    },
    'binary_labels': {
        'input': Path('data/cache/step_4a_threshold'),
        'output': 'labels/binary'
    },
    'colored_labels': {
        'input': Path('data/cache/step_4a_threshold'),
        'output': 'labels/colored'
    }
}

default_settings = {
    'frame_info_csv': Path('data/frame_info.csv'),
    'output_base_dir': Path('data/output/images'),
    'steps': ['original', 'filtered', 'whitened', 'denoised', 'binary_labels', 'colored_labels'],
    'image_format': 'png',
    'dpi': 100,
    'figsize': (12, 3),
    'cmap': 'gist_heat',
    'quantiles': [0.1, 0.99],
    'aspect': 'auto',
    'origin': 'lower',
    'nfft': 1024,
    'overwrite': False,
}


def get_extent(frame_info: pd.DataFrame, file_indices: list[int], data_shape: tuple, nfft: int) -> list:
    """Calculate extent from frame_info CSV and data shape."""
    F, T = data_shape
    time_start = frame_info.loc[file_indices[0], 'time_start']
    time_end = frame_info.loc[file_indices[-1], 'time_end']
    freq_end = (F / nfft * 500)  # Nyquist at 500 kHz for 1 MHz sample rate
    return [time_start, time_end, 0, freq_end]


def save_plot(data: np.ndarray, output_path: Path, extent: list, title: str, settings: dict, plot_type: str = 'spectrogram') -> None:
    """Unified function to save all plot types."""
    fig = plt.figure(figsize=settings['figsize'], dpi=settings['dpi'])
    
    if plot_type == 'binary':
        plt.imshow(data, aspect=settings['aspect'], origin=settings['origin'], 
                   cmap='gray', vmin=0, vmax=1, extent=extent)
    elif plot_type == 'labeled':
        num_features = int(data.max())
        colors = plt.cm.tab20(np.linspace(0, 1, min(num_features, 20))) if num_features <= 20 else plt.cm.hsv(np.linspace(0, 1, num_features))
        cmap = mcolors.ListedColormap(['black'] + list(colors))
        norm = mcolors.BoundaryNorm(np.arange(-0.5, num_features + 1.5, 1), cmap.N)
        plt.imshow(data, aspect=settings['aspect'], origin=settings['origin'], 
                   cmap=cmap, norm=norm, extent=extent)
    else:  # spectrogram
        vmin, vmax = np.quantile(data, settings['quantiles'])
        plt.imshow(data, aspect=settings['aspect'], origin=settings['origin'], 
                   cmap=settings['cmap'], vmin=vmin, vmax=vmax, extent=extent)
    
    plt.title(title, fontweight='bold')
    plt.xlabel('Time [ms]', fontweight='bold')
    plt.ylabel('Frequency [kHz]', fontweight='bold')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def process_data(channel_data: np.ndarray, step_name: str) -> np.ndarray:
    """Process channel data based on step type."""
    if step_name in ['binary_labels', 'colored_labels']:
        mask = channel_data.squeeze(axis=-1)
        return label(mask)[0] if step_name == 'colored_labels' else mask
    
    # Process complex spectrogram data
    if step_name == 'denoised':
        complex_data = channel_data[..., 2] + -1j * channel_data[..., 3]
    else:
        complex_data = channel_data[..., 0] + 1j * channel_data[..., 1]
    
    abs_data = np.abs(complex_data)
    if abs_data.max() > 0:
        complex_data = complex_data / abs_data.max()
    return np.log1p(np.abs(complex_data))


def process_single(shotn: int, frame_info: pd.DataFrame, settings: dict, step_name: str, step_config: dict) -> None:
    """Process a single shot and generate images for all channels."""
    # Get file indices
    file_indices = frame_info[frame_info['shotn'] == shotn].index.tolist()
    if not file_indices:
        logger.warning(f"No files found for shot {shotn}")
        return
    
    # Load and concatenate data
    try:
        data_list = [joblib.load(step_config['input'] / f'{idx}.joblib') for idx in file_indices 
                     if (step_config['input'] / f'{idx}.joblib').exists()]
        if not data_list:
            raise ValueError(f"No data found for shot {shotn}")
        data = np.concatenate(data_list, axis=2)
    except Exception as e:
        logger.error(f"Error loading data for shot {shotn}: {e}")
        return
    
    # Setup output directory
    output_dir = settings['output_base_dir'] / step_config['output']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each channel
    C = data.shape[0]
    for channel_idx in range(C):
        try:
            # Process data
            processed = process_data(data[channel_idx], step_name)
            
            # Get extent from frame_info
            extent = get_extent(frame_info, file_indices, processed.shape, settings['nfft'])
            
            # Determine plot type
            plot_type = 'binary' if step_name == 'binary_labels' else 'labeled' if step_name == 'colored_labels' else 'spectrogram'
            
            # Save plot
            output_path = output_dir / f"{shotn}_channel_{channel_idx}.{settings['image_format']}"
            title = f"Shot {shotn} - Channel {channel_idx}"
            save_plot(processed, output_path, extent, title, settings, plot_type)
            
        except Exception as e:
            logger.error(f"Error processing shot {shotn}, channel {channel_idx}: {e}")


def main(config_path: Path | str | None = None) -> None:
    """Main function to generate images from spectrogram data."""
    settings = load_settings(config_path, default_settings)
    frame_info = pd.read_csv(settings['frame_info_csv'])
    shot_numbers = frame_info['shotn'].unique().tolist()
    logger.info(f"Found {len(shot_numbers)} unique shots")
    
    setup_directory(path=settings['output_base_dir'], overwrite=settings['overwrite'])
    
    for step_name in settings['steps']:
        if step_name not in STEP_MAPPINGS:
            logger.warning(f"Unknown step: {step_name}, skipping")
            continue
        
        step_config = STEP_MAPPINGS[step_name]
        if not step_config['input'].exists():
            logger.warning(f"Input directory not found: {step_config['input']}, skipping")
            continue
        
        logger.info(f"Processing step: {step_name}")
        mapper = ParallelMapper()
        mapper(process_single, shot_numbers, frame_info=frame_info, 
               settings=settings, step_name=step_name, step_config=step_config)
    
    logger.info("Image generation complete!")


if __name__ == '__main__':
    # python -m autotslabel.autosegment.multichannel.utils.generate_images
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path)
