import sys
import numpy as np
import pandas as pd
from pathlib import Path

import joblib
from tqdm.auto import tqdm

import logging
logger = logging.getLogger(__name__)

from .utils.configuration import (
    load_settings, 
    setup_directory,
)

default_settings = {
    'frame_info_csv': Path('data/frame_info.csv'),
    'input_dir_img': Path('data/cache/original/step_2a_make_spectrogram'),
    'input_dir_mask': Path('data/cache/step_4a_threshold'),
    'output_dir': Path('data/cache/step_5a_combine_spectrogram'),
    'overwrite': True,
}

def load_frame_info(csv_path: Path) -> pd.DataFrame:
    """Load and parse frame_info.csv."""
    df = pd.read_csv(csv_path)
    return df


def get_shot_files(shotn: int, frame_info: pd.DataFrame) -> list[int]:
    """Get list of file indices for a given shot."""
    shot_df = frame_info[frame_info['shotn'] == shotn]
    # File indices are row indices in the CSV
    return shot_df.index.tolist()


def combine_shot_all_channels(
    file_indices: list[int],
    input_dir: Path,
    output_dir: Path,
    shotn: int,
    method: str = 'img',
    overwrite: bool = True,
) -> None:
    """
    Combine time windows for all channels of a shot at once.
    
    Args:
        file_indices: List of file indices to concatenate
        input_dir: Directory containing input joblib files
        output_dir: Directory to save combined outputs
        shotn: Shot number
        method: 'img' for spectrograms (4D) or 'mask' for masks (4D)
        overwrite: Whether to overwrite existing files
    """
    # Load all files for this shot
    all_data = []
    
    for idx in file_indices:
        file_path = input_dir / f'{idx}.joblib'
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            continue
        
        data = joblib.load(file_path)
        all_data.append(data)
    
    if len(all_data) == 0:
        raise ValueError(f"No data found for file indices {file_indices}")
    
    # Get number of channels from first file
    num_channels = all_data[0].shape[0]
    
    # For each channel, extract and concatenate
    for channel_idx in range(num_channels):
        # Determine output path
        name_ext = '_mask' if method == 'mask' else ''
        output_path = output_dir / f'{shotn}_{channel_idx}{name_ext}.joblib'
        
        if output_path.exists() and not overwrite:
            logger.debug(f"Skipping existing file: {output_path.name}")
            continue
        
        # Extract this channel from all files
        # For img: data shape is (C, F, W, 2)
        # For mask: data shape is (C, H, W, 1)
        channel_data_list = [data[channel_idx] for data in all_data]  # List of (F, W, 2) or (H, W, 1)
        
        # Concatenate along width axis (axis=1)
        # Result: (F, W_total, 2) or (H, W_total, 1)
        concatenated = np.concatenate(channel_data_list, axis=1)
        
        # For images, convert complex to magnitude
        if method == 'img':
            # Complex data has shape (F, W_total, 2) where [..., 0] is real, [..., 1] is imaginary
            complex_data = concatenated[..., 0] + 1j * concatenated[..., 1]
            magnitude = np.abs(complex_data)
            # Apply log transform (no normalization)
            concatenated = np.log1p(magnitude)  # Shape: (F, W_total)
        else:
            # For masks, just squeeze the last dimension
            concatenated = concatenated.squeeze(axis=-1)  # Shape: (H, W_total)
        
        # Save combined data
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(concatenated, output_path, compress=True)
        logger.debug(f"Saved {output_path.name} with shape {concatenated.shape}")


def process_shot(
    shotn: int,
    file_indices: list[int],
    settings: dict,
) -> None:
    """
    Process all channels for a single shot.
    
    Args:
        shotn: Shot number
        file_indices: List of file indices for this shot
        settings: Settings dictionary
    """
    logger.info(f"Processing shot {shotn} with {len(file_indices)} files")
    
    # Process all image channels at once
    try:
        combine_shot_all_channels(
            file_indices=file_indices,
            input_dir=settings['input_dir_img'],
            output_dir=settings['output_dir'],
            shotn=shotn,
            method='img',
            overwrite=settings['overwrite'],
        )
    except Exception as e:
        logger.error(f"Error processing shot {shotn} (img): {e}")
    
    # Process all mask channels at once
    try:
        combine_shot_all_channels(
            file_indices=file_indices,
            input_dir=settings['input_dir_mask'],
            output_dir=settings['output_dir'],
            shotn=shotn,
            method='mask',
            overwrite=settings['overwrite'],
        )
    except Exception as e:
        logger.error(f"Error processing shot {shotn} (mask): {e}")


def main(config_path: Path | str | None = None) -> None:
    """Main function to combine time windows into full shot files."""
    settings = load_settings(config_path, default_settings)
    
    # Setup output directory
    setup_directory(
        path=settings['output_dir'], 
        overwrite=settings['overwrite'],
    )
    
    # Load frame info
    logger.info(f"Loading frame info from {settings['frame_info_csv']}")
    frame_info = load_frame_info(settings['frame_info_csv'])
    
    # Get unique shot numbers
    shot_numbers = frame_info['shotn'].unique().tolist()
    logger.info(f"Found {len(shot_numbers)} unique shots")
    
    # Process each shot
    for shotn in tqdm(shot_numbers, desc="Combining shots"):
        # Get file indices for this shot
        file_indices = get_shot_files(shotn, frame_info)
        if len(file_indices) == 0:
            logger.warning(f"No files found for shot {shotn}")
            continue
        
        # Process this shot
        process_shot(shotn, file_indices, settings)
    
    logger.info("Combination complete!")


if __name__ == '__main__':
    # python -m autotslabel.autosegment.multichannel.step_5a_combine_spectrogram
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path)
