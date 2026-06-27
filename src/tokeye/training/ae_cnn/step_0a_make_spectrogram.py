"""
Step 0a: Compute spectrograms from raw HDF5 signals and save as multichannel TIFFs.

This step:
1. Loads raw HDF5 signal data for each shot
2. Resamples from ~1666 Hz to 500 Hz
3. Computes STFT spectrograms with log1p transform
4. Saves 4-channel spectrograms (r0, v1, v2, v3) as TIFFs
5. Computes and saves global mean/std statistics

Usage:
    python -m aemodes.pipeline.step_0a_make_spectrogram
"""

import numpy as np
import tifffile as tif
import pandas as pd
import json

from pathlib import Path
import shutil

from scipy import signal

from tqdm.auto import tqdm

from aemodes.utils.dataset import load_dataset

default_settings = {
    "diagnostic": "co2_density",
    "time_range": (0, 2000),
    "channels": ["r0", "v1", "v2", "v3"],
    "up_factor": 3,
    "down_factor": 10,
    "nperseg": 1024,
    "hop": 128,
    "data_dirs": [
        Path('/scratch/gpfs/EKOLEMEN/d3d_fusion_data/'),
        Path('/scratch/gpfs/EKOLEMEN/hackathon/raw_h5_files/')
    ],
    "label_path": "data/co2_250_detector.pkl",
    "output_path": "data/.cache/step_0a",
}


def compute_spectrogram(data, fs, settings):
    """
    Compute spectrogram from raw signal data.
    
    1. Resample from ~1666 Hz to 500 Hz using resample_poly
    2. Compute STFT using ShortTimeFFT
    3. Apply log1p transform
    
    Args:
        data: Raw signal data (1D numpy array)
        fs: Original sampling rate
        settings: Settings dict with up_factor, down_factor, nperseg, hop
    
    Returns:
        Sxx: Log-transformed spectrogram (2D numpy array)
        new_fs: New sampling rate after resampling
    """
    up_factor = settings["up_factor"]
    down_factor = settings["down_factor"]
    nperseg = settings["nperseg"]
    hop = settings["hop"]
    
    # Resample to ~500 Hz
    resampled_data = signal.resample_poly(data, up_factor, down_factor)
    new_fs = fs * up_factor / down_factor
    
    # Create ShortTimeFFT object and compute spectrogram
    win = signal.get_window('hann', nperseg)
    SFT = signal.ShortTimeFFT(win, hop=hop, fs=new_fs)
    Sxx = SFT.spectrogram(resampled_data)[1:]  # Skip DC component
    Sxx = np.log1p(Sxx)
    
    return Sxx, new_fs


def find_shot_paths(shots, data_dirs):
    """
    Find HDF5 file paths for given shots.
    
    Args:
        shots: List of shot numbers
        data_dirs: List of directories to search
    
    Returns:
        Dict mapping shot number to file path
    """
    shot_paths = {}
    for shot in shots:
        for data_dir in data_dirs:
            file_path = data_dir / f'{shot}.h5'
            if file_path.exists():
                shot_paths[shot] = file_path
                break
    return shot_paths


def make_spectrograms(shots, shot_paths, settings, mode='train'):
    """
    Compute and save spectrograms for all shots.
    
    Args:
        shots: List of shot numbers
        shot_paths: Dict mapping shot numbers to HDF5 file paths
        settings: Settings dict
        mode: 'train' or 'valid'
    
    Returns:
        running_sum: Sum of all spectrogram values
        running_sum_sq: Sum of squared spectrogram values
        total_count: Total number of values
    """
    diagnostic = settings["diagnostic"]
    time_range = settings["time_range"]
    channels = settings["channels"]
    output_path = Path(settings["output_path"]) / "spectrograms"
    
    # Running statistics accumulators
    running_sum = 0.0
    running_sum_sq = 0.0
    total_count = 0
    
    for shot in tqdm(shots, desc=f"Computing spectrograms ({mode})"):
        if shot not in shot_paths:
            print(f"Warning: Shot {shot} not found in data directories, skipping")
            continue
        
        # Load raw HDF5 data
        file_path = shot_paths[shot]
        with pd.HDFStore(file_path) as store:
            data = store[diagnostic]
        data = data.loc[time_range[0]:time_range[1]]
        fs = len(data) / (data.index[-1] - data.index[0])
        
        # Compute spectrogram for each channel
        channel_spectrograms = []
        for channel in channels:
            channel_data = data[channel].values
            Sxx, _ = compute_spectrogram(channel_data, fs, settings)
            channel_spectrograms.append(Sxx)
        
        # Stack channels: shape (4, freq, time)
        multichannel_spectrogram = np.stack(channel_spectrograms, axis=0).astype(np.float32)
        
        # Accumulate statistics
        running_sum += multichannel_spectrogram.sum()
        running_sum_sq += (multichannel_spectrogram ** 2).sum()
        total_count += multichannel_spectrogram.size
        
        # Save multichannel TIF
        tif.imwrite(
            output_path / f"{shot}_{mode}.tif",
            multichannel_spectrogram
        )
    
    return running_sum, running_sum_sq, total_count


if __name__ == '__main__':
    # python -m aemodes.pipeline.step_0a_make_spectrogram
    
    settings = default_settings
    
    # Create output directories
    output_path = Path(settings['output_path'])
    spectrogram_path = output_path / 'spectrograms'
    if spectrogram_path.exists():
        shutil.rmtree(spectrogram_path)
    spectrogram_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset (for shot list only)
    train_dataset, valid_dataset = load_dataset(settings['label_path'])
    train_shots = train_dataset.shots
    valid_shots = valid_dataset.shots
    print(f"Found {len(train_shots)} train shots, {len(valid_shots)} valid shots")
    
    # Find available shot paths
    train_shot_paths = find_shot_paths(train_shots, settings['data_dirs'])
    valid_shot_paths = find_shot_paths(valid_shots, settings['data_dirs'])
    print(f"Available: {len(train_shot_paths)} train shots, {len(valid_shot_paths)} valid shots")
    
    # Compute and save spectrograms, accumulating statistics
    train_sum, train_sum_sq, train_count = make_spectrograms(
        train_shots, train_shot_paths, settings, mode='train'
    )
    valid_sum, valid_sum_sq, valid_count = make_spectrograms(
        valid_shots, valid_shot_paths, settings, mode='valid'
    )
    
    # Compute global mean and std
    total_sum = train_sum + valid_sum
    total_sum_sq = train_sum_sq + valid_sum_sq
    total_count = train_count + valid_count
    
    global_mean = total_sum / total_count
    global_std = np.sqrt(total_sum_sq / total_count - global_mean ** 2)
    
    # Save statistics
    stats = {
        "mean": float(global_mean),
        "std": float(global_std),
        "total_count": int(total_count),
    }
    
    stats_path = output_path / "stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nStatistics saved to {stats_path}")
    print(f"  Mean: {global_mean:.6f}")
    print(f"  Std:  {global_std:.6f}")
    print(f"  Total values: {total_count:,}")
    
    print("\nStep 0a completed")

