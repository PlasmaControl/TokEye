import sys
import numpy as np
import pandas as pd
from pathlib import Path

from tqdm.auto import tqdm

import joblib

from skimage.morphology import remove_small_objects

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .utils.configuration import (
    load_settings, 
    load_input_paths,
    setup_directory,
)
from .utils.parmap import ParallelMapper

default_settings = {
    'min_size': 64,
    'remove_bottom_rows': 5,
    'remove_top_rows': 2,
    'frame_info_csv': Path('data/library/bes_250/frame_info.csv'),
    'threshold_output_path': Path('data/library/bes_250/thresholds_baseline.txt'),
    'input_dir': Path('data/library/bes_250/step_2b_filter_spectrogram_baseline'),
    'output_dir': Path('data/library/bes_250/step_4a_threshold_baseline'),
    'overwrite': True,
}

def get_threshold(data, adjust=0., multiplier=100):
    """
    Compute threshold using triangle method on cumulative distribution.
    
    Args:
        data: 2D array (H, W)
        adjust: adjustment factor for threshold
        multiplier: resolution multiplier for interpolation
    
    Returns:
        threshold value
    """
    H, W = data.shape
    median = np.median(data)
    data_2 = data.copy()
    data_2[data < median] = median

    sorted_data = np.sort(data_2.flatten())
    data_min = sorted_data.min()
    minmax = sorted_data.max() - sorted_data.min()
    sorted_data = (sorted_data - sorted_data.min()) / minmax * multiplier
    cdf_values = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    cdf_values = (cdf_values - cdf_values.min()) / (cdf_values.max() - cdf_values.min()) * multiplier * 2

    x_cdf = np.linspace(sorted_data.min(), sorted_data.max(), multiplier)
    cdf = np.interp(x_cdf, sorted_data, cdf_values)

    start_x, end_x = x_cdf[0], x_cdf[-1]
    start_y, end_y = cdf[0], cdf[-1]

    triangle_slope = (end_y - start_y) / (end_x - start_x)
    triangle_intercept = start_y - triangle_slope * start_x

    a, b, c = -triangle_slope, 1, -triangle_intercept

    distances = np.abs(a * x_cdf + b * cdf + c) / np.sqrt(a**2 + b**2)
    threshold_idx = np.argmax(distances)
    binary = x_cdf[threshold_idx]
    binary = binary / multiplier * minmax + data_min
    binary = binary + adjust * (data_2.max() - data_2.min())

    return binary


def load_frame_info(csv_path: Path) -> pd.DataFrame:
    """Load and parse frame_info.csv."""
    df = pd.read_csv(csv_path)
    return df


def get_shot_files(shotn: int, frame_info: pd.DataFrame) -> list[int]:
    """Get list of file indices for a given shot."""
    shot_df = frame_info[frame_info['shotn'] == shotn]
    # File indices are row indices in the CSV
    return shot_df.index.tolist()


def load_and_concatenate_shot(file_indices: list[int], input_dir: Path) -> np.ndarray:
    """Load all joblib files for the shot and concatenate along time axis."""
    data_list = []
    for idx in file_indices:
        file_path = input_dir / f'{idx}.joblib'
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            continue
        data = joblib.load(file_path)
        data_list.append(data)
    
    if len(data_list) == 0:
        raise ValueError(f"No data found for file indices {file_indices}")
    
    # Concatenate along time axis (axis=2)
    concatenated = np.concatenate(data_list, axis=2)
    return concatenated


def compute_thresholds_for_shots(
    frame_info: pd.DataFrame,
    input_dir: Path,
    settings: dict,
) -> pd.DataFrame:
    """
    Compute thresholds for each (shot, channel) pair.
    
    Args:
        frame_info: DataFrame with frame information
        input_dir: Directory containing input joblib files
        settings: Settings dict with processing parameters
    
    Returns:
        DataFrame with columns: shotn, channel, threshold
    """
    logger.info("Computing thresholds for all shots...")
    
    # Get unique shot numbers
    shot_numbers = frame_info['shotn'].unique().tolist()
    logger.info(f"Found {len(shot_numbers)} unique shots")
    
    threshold_data = []
    
    for shotn in tqdm(shot_numbers, desc="Computing thresholds"):
        try:
            # Get file indices for this shot
            file_indices = get_shot_files(shotn, frame_info)
            if len(file_indices) == 0:
                logger.warning(f"No files found for shot {shotn}")
                continue
            
            # Load and concatenate data
            data = load_and_concatenate_shot(file_indices, input_dir)
            C, H, W, Z = data.shape
            
            # Process each channel
            for channel_idx in range(C):
                # Extract channel data and average across Z dimension
                channel_data = data[channel_idx]  # Shape: (H, W, Z)
                channel_data_mean = np.sqrt(np.mean(channel_data**2, axis=-1))  # Shape: (H, W)
                
                # Apply masking to top/bottom rows
                channel_data_mean[:settings['remove_bottom_rows'], :] = channel_data_mean.mean()
                channel_data_mean[-settings['remove_top_rows']:, :] = channel_data_mean.mean()
                channel_data_mean[:,:3] = channel_data_mean.mean()
                # Compute threshold

                # for baseline specifically
                channel_data_mean = np.gradient(channel_data_mean, axis=1)
                # end

                threshold = get_threshold(channel_data_mean, adjust=0.05) # adjust for 
                
                # Store result
                threshold_data.append({
                    'shotn': shotn,
                    'channel': channel_idx,
                    'threshold': threshold,
                })
        
        except Exception as e:
            logger.error(f"Error computing threshold for shot {shotn}: {e}")
            continue
    
    # Create DataFrame
    threshold_df = pd.DataFrame(threshold_data)
    return threshold_df


def load_thresholds(threshold_path: Path) -> dict:
    """
    Load thresholds from CSV file into lookup dictionary.
    
    Args:
        threshold_path: Path to threshold CSV file
    
    Returns:
        Dictionary mapping (shotn, channel) -> threshold
    """
    df = pd.read_csv(threshold_path)
    threshold_dict = {}
    for _, row in df.iterrows():
        key = (int(row['shotn']), int(row['channel']))
        threshold_dict[key] = row['threshold']
    return threshold_dict


def process_single(
    input_path: Path,
    settings: dict,
    threshold_dict: dict,
    frame_info: pd.DataFrame,
) -> None:
    """
    Process a single file by applying precomputed thresholds.
    
    Args:
        input_path: Path to input joblib file
        output_dir: Directory to save output
        settings: Settings dict with processing parameters
        threshold_dict: Dictionary mapping (shotn, channel) -> threshold
        frame_info: DataFrame with frame information
    """
    output_path = settings['output_dir'] / input_path.name
    if output_path.exists() and not settings['overwrite']:
        return
    
    try:
        # Load data
        data = joblib.load(input_path)
        C, H, W, Z = data.shape
        
        # Get file index from filename
        file_idx = int(input_path.stem)
        
        # Get shot number for this file
        if file_idx >= len(frame_info):
            logger.error(f"File index {file_idx} out of range for frame_info")
            return
        
        shotn = frame_info.iloc[file_idx]['shotn']
        
        # Process each channel
        output_data = np.zeros((C, H, W, 1), dtype=bool)
        
        for channel_idx in range(C):
            # Look up threshold
            key = (shotn, channel_idx)
            if key not in threshold_dict:
                logger.warning(f"Threshold not found for shot {shotn}, channel {channel_idx}")
                continue
            
            threshold = threshold_dict[key]
            
            # Extract channel data and compute RMS across Z
            channel_data = data[channel_idx]  # Shape: (H, W, Z)
            channel_data_mean = np.sqrt(np.mean(channel_data**2, axis=-1))  # Shape: (H, W)
            channel_data_mean = np.gradient(channel_data_mean, axis=1)
            
            # Apply threshold
            binary = channel_data_mean > threshold

            # Apply masking to top/bottom rows
            binary[:settings['remove_bottom_rows'], :] = 0
            binary[-settings['remove_top_rows']:, :] = 0
            binary[:,-1:] = 0
            binary[:,:2] = 0
            
            # Remove small objects
            binary = remove_small_objects(binary, min_size=settings['min_size'])
            
            # Store result with expanded dimension
            output_data[channel_idx] = np.expand_dims(binary, axis=-1)
        
        # Save output
        joblib.dump(output_data, output_path)
    
    except Exception as e:
        logger.error(f"Error processing file {input_path}: {e}")


def main(config_path: Path | str | None = None) -> None:
    """Main function to compute thresholds and apply them to individual files."""
    settings = load_settings(config_path, default_settings)
    
    # Setup output directory
    setup_directory(
        path=settings['output_dir'], 
        overwrite=settings['overwrite'],
    )
    
    # Load frame info
    frame_info = load_frame_info(settings['frame_info_csv'])
    
    # Phase 1: Compute thresholds for all shots
    logger.info("Phase 1: Computing thresholds...")
    threshold_df = compute_thresholds_for_shots(
        frame_info,
        settings['input_dir'],
        settings,
    )
    
    # Save thresholds to file
    threshold_df.to_csv(settings['threshold_output_path'], index=False)
    logger.info(f"Saved thresholds to {settings['threshold_output_path']}")
    
    # Phase 2: Apply thresholds to individual files
    logger.info("Phase 2: Applying thresholds to individual files...")
    
    # Load thresholds into dictionary
    threshold_dict = load_thresholds(settings['threshold_output_path'])
    
    # Get input paths
    input_paths = load_input_paths(settings['input_dir'])
    
    # Process all files using parallel mapper
    mapper = ParallelMapper()
    mapper(
        process_single,
        input_paths, 
        settings=settings,
        threshold_dict=threshold_dict,
        frame_info=frame_info,
    )
    
    logger.info("Threshold application complete!")


if __name__ == "__main__":
    # python -m autotslabel.autosegment.multichannel.step_4a_threshold
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path)
