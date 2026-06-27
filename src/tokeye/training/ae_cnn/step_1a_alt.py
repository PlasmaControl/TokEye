"""
Step 1a (Alternative): Generate semantic labels using pre-computed spectrograms.

This step:
1. Loads pre-computed multichannel spectrograms from step_0a
2. Normalizes using global mean/std from stats.json
3. Runs BigTFUNetModel for mode detection
4. Creates overlap labels with ground truth
5. Saves input/label TIFFs for training

Usage:
    python -m aemodes.pipeline.step_1a_alt
"""

import numpy as np
import tifffile as tif
import json

from pathlib import Path
import shutil

from scipy.ndimage import zoom

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from tqdm.auto import tqdm

from tokeye.models.big_tf_unet.model_big_tf_unet import BigTFUNetModel
from tokeye.models.big_tf_unet.config_big_tf_unet import BigTFUNetConfig

from aemodes.utils.dataset import load_dataset

default_settings = {
    "threshold": 0.5,
    "channels": ["r0", "v1", "v2", "v3"],
    "label_path": "data/co2_250_detector.pkl",
    "spectrogram_path": "data/.cache/step_0a/spectrograms",
    "stats_path": "data/.cache/step_0a/stats.json",
    "model_path": "/scratch/gpfs/nc1514/tokeye/model/big_tf_unet_251210.pt",
    "output_path": "data/.cache/step_1a_alt",
}


def load_stats(stats_path):
    """
    Load global normalization statistics.
    
    Args:
        stats_path: Path to stats.json
    
    Returns:
        mean: Global mean
        std: Global standard deviation
    """
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    return stats["mean"], stats["std"]


def make_semantic_alt(model, dataset, settings, mean, std, mode='train'):
    """
    Generate semantic labels using pre-computed spectrograms.
    
    Args:
        model: BigTFUNetModel for mode detection
        dataset: ShotDataset with shots, ground truth labels
        settings: Settings dict
        mean: Global mean for normalization
        std: Global std for normalization
        mode: 'train' or 'valid'
    """
    channels = settings["channels"]
    threshold = settings["threshold"]
    output_path = settings["output_path"]
    spectrogram_path = Path(settings["spectrogram_path"])
    
    # Cache for loaded spectrograms (avoid reloading for each window)
    spectrogram_cache = {}
    
    for idx in tqdm(range(len(dataset)), desc=f"Processing {mode}"):
        # Get shot info from dataset
        sample = dataset[idx]
        shot_idx = idx // dataset.nwin
        win_idx = idx % dataset.nwin
        shot = dataset.shots[shot_idx]
        
        # Load spectrogram from cache or file
        if shot not in spectrogram_cache:
            spectrogram_file = spectrogram_path / f"{shot}_{mode}.tif"
            if not spectrogram_file.exists():
                print(f"Warning: Spectrogram for shot {shot} not found, skipping")
                continue
            spectrogram_cache[shot] = tif.imread(spectrogram_file)
        
        multichannel_spectrogram = spectrogram_cache[shot]
        
        # Get ground truth label for this window
        ae_true = sample['y'].numpy().sum(axis=0) > 0
        ae_true = ae_true[None, :]
        
        # Compute window indices based on spectrogram dimensions
        lenshot = multichannel_spectrogram.shape[2]  # shape is (4, freq, time)
        lenwin = lenshot // dataset.nwin
        hoplen = lenshot // dataset.nwin
        start_idx = win_idx * hoplen
        end_idx = start_idx + lenwin
        
        # Process each channel
        for i, channel in enumerate(channels):
            # Extract window from pre-computed spectrogram
            Sxx_window = multichannel_spectrogram[i, :, start_idx:end_idx]
            
            # Apply global normalization
            inp = torch.from_numpy(Sxx_window).float()
            inp = inp.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            inp = (inp - mean) / std
            inp = inp.to(device)
            
            # Run model
            with torch.no_grad():
                out = model(inp)[0]
                out = torch.sigmoid(out)
            out = out > threshold
            out = out.squeeze(0)[0].cpu().numpy()
            
            # Create overlap label with ground truth
            # Resize ground truth to match spectrogram dimensions if needed
            if out.shape[1] != ae_true.shape[1]:
                # Interpolate ae_true to match spectrogram time dimension
                zoom_factor = out.shape[1] / ae_true.shape[1]
                ae_true_resized = zoom(ae_true.astype(float), (1, zoom_factor), order=0) > 0.5
            else:
                ae_true_resized = ae_true
            
            overlap = out * ae_true_resized
            
            # Input array (normalized spectrogram window)
            inp_array = inp.squeeze(0).squeeze(0).cpu().numpy()
            out_array = overlap.astype(np.float32)
            
            # Save files
            tif.imwrite(
                f"{output_path}/input/{idx}_{i}_{mode}.tif",
                inp_array.astype(np.float32)
            )
            tif.imwrite(
                f"{output_path}/label/{idx}_{i}_{mode}.tif",
                out_array
            )
        
        # Clear cache periodically to save memory (keep only last few shots)
        if len(spectrogram_cache) > 10:
            oldest_shot = next(iter(spectrogram_cache))
            del spectrogram_cache[oldest_shot]


if __name__ == '__main__':
    # python -m aemodes.pipeline.step_1a_alt
    
    settings = default_settings
    
    # Check that step_0a has been run
    stats_path = Path(settings['stats_path'])
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Stats file not found at {stats_path}. "
            "Please run step_0a_make_spectrogram.py first."
        )
    
    # Load global normalization statistics
    mean, std = load_stats(settings['stats_path'])
    print(f"Loaded stats: mean={mean:.6f}, std={std:.6f}")
    
    # Create output directories
    input_output_path = Path(settings['output_path']) / 'input'
    label_output_path = Path(settings['output_path']) / 'label'
    if input_output_path.exists():
        shutil.rmtree(input_output_path)
    if label_output_path.exists():
        shutil.rmtree(label_output_path)
    input_output_path.mkdir(parents=True, exist_ok=True)
    label_output_path.mkdir(parents=True, exist_ok=True)
    
    # Load BigTFUNetModel
    config = BigTFUNetConfig()
    state_dict = torch.load(settings['model_path'], weights_only=True)
    model = BigTFUNetModel(config)
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device)
    print("Model loaded")
    
    # Load dataset (for shot list and ground truth labels)
    train_dataset, valid_dataset = load_dataset(settings['label_path'])
    print(f"Loaded {len(train_dataset)} train samples, {len(valid_dataset)} valid samples")
    
    # Generate semantic labels
    make_semantic_alt(model, train_dataset, settings, mean, std, mode='train')
    make_semantic_alt(model, valid_dataset, settings, mean, std, mode='valid')
    
    print("Step 1a (alternative) completed")
