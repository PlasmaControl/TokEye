import sys
import numpy as np
from pathlib import Path

import joblib
import tifffile as tif
from tqdm.auto import tqdm

import logging
logger = logging.getLogger(__name__)

from .utils.configuration import (
    load_settings, 
    load_input_paths,
    setup_directory,
)
from .utils.parmap import ParallelMapper

default_settings = {
    'output_channel': 'single',
    'zscore_clip': 3,
    'train_dirs_file': Path('data/settings/train_dirs.txt'),
    'output_dir': Path('data/cache/step_6a_convert_tif'),
    'overwrite': True,
}

def load_train_directories(train_dirs_file: Path) -> list[Path]:
    """Load training directories from file."""
    if not train_dirs_file.exists():
        logger.error(f"Train directories file not found: {train_dirs_file}")
        return []
    
    directories = []
    with open(train_dirs_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                directories.append(Path(line))
    
    logger.info(f"Loaded {len(directories)} training directories from {train_dirs_file}")
    return directories

def validate_directory(base_dir: Path) -> tuple[bool, Path | None, Path | None, Path | None, list[Path], list[Path], list[Path]]:
    """
    Validate directory has both input and label data (normal and baseline).
    
    Returns:
        tuple: (is_valid, input_dir, label_dir, label_dir_baseline, input_paths, label_paths, label_paths_baseline)
    """
    input_dir = base_dir / 'original' / 'step_2a_make_spectrogram'
    label_dir = base_dir / 'step_4a_threshold'
    label_dir_baseline = base_dir / 'step_4a_threshold_baseline'
    
    # Check if directories exist
    if not input_dir.exists():
        logger.warning(f"Input directory does not exist: {input_dir}")
        return False, None, None, None, [], [], []
    
    if not label_dir.exists():
        logger.warning(f"Label directory does not exist: {label_dir}")
        return False, None, None, None, [], [], []
    
    if not label_dir_baseline.exists():
        logger.warning(f"Baseline label directory does not exist: {label_dir_baseline}")
        return False, None, None, None, [], [], []
    
    # Check for .joblib files
    input_paths = load_input_paths(input_dir)
    label_paths = load_input_paths(label_dir)
    label_paths_baseline = load_input_paths(label_dir_baseline)
    
    if len(input_paths) == 0:
        logger.warning(f"No input .joblib files found in {input_dir}")
        return False, None, None, None, [], [], []
    
    if len(label_paths) == 0:
        logger.warning(f"No label .joblib files found in {label_dir}")
        return False, None, None, None, [], [], []
    
    if len(label_paths_baseline) == 0:
        logger.warning(f"No baseline label .joblib files found in {label_dir_baseline}")
        return False, None, None, None, [], [], []
    
    logger.info(f"Valid directory: {base_dir} ({len(input_paths)} inputs, {len(label_paths)} labels, {len(label_paths_baseline)} baseline labels)")
    return True, input_dir, label_dir, label_dir_baseline, input_paths, label_paths, label_paths_baseline

def compute_single_file_statistics(
    input_path: Path,
) -> dict:
    """Compute statistics for a single file."""
    
    data = joblib.load(input_path)
    C = data.shape[0]
    
    channel_sums = np.zeros(C)
    channel_sq_sums = np.zeros(C)
    channel_counts = np.zeros(C)
    
    for i in range(C):
        magnitude = np.sqrt(data[i, ..., 0]**2 + data[i, ..., 1]**2).astype(np.float32)
        magnitude = np.log1p(magnitude)
        channel_sums[i] = magnitude.sum()
        channel_sq_sums[i] = (magnitude**2).sum()
        channel_counts[i] = magnitude.size
    
    return {
        'sums': channel_sums,
        'sq_sums': channel_sq_sums,
        'counts': channel_counts,
    }

def collect_image_statistics(
    input_paths: list[Path],
) -> dict:
    """Collect global statistics (mean, std) per channel across all images."""
    
    logger.info("Collecting image statistics...")
    
    # Use parallel processing to compute statistics
    mapper = ParallelMapper()
    results = mapper(compute_single_file_statistics, input_paths)
    
    # Determine number of channels from first result
    C = len(results[0]['sums'])
    
    # Aggregate statistics across all files
    channel_sums = np.zeros(C)
    channel_sq_sums = np.zeros(C)
    channel_counts = np.zeros(C)
    
    for result in tqdm(results, desc="Computing statistics"):
        channel_sums += result['sums']
        channel_sq_sums += result['sq_sums']
        channel_counts += result['counts']
    
    # Compute global mean and std per channel
    channel_means = channel_sums / channel_counts
    channel_stds = np.sqrt(channel_sq_sums / channel_counts - channel_means**2)
    
    logger.info(f"Statistics computed: {C} channels")
    
    return {
        'means': channel_means,
        'stds': channel_stds,
    }

def process_data_img(
    data: np.ndarray,
    channel_idx: int,
    stats: dict,
    zscore_clip: float,
) -> np.ndarray:
    """Process image data: compute magnitude, log1p, clip, standardize, normalize."""
    
    # Compute magnitude from real and imaginary parts
    magnitude = np.sqrt(data[..., 0]**2 + data[..., 1]**2).astype(np.float32)
    magnitude = np.log1p(magnitude)
    
    # Get statistics for this channel
    mean = stats['means'][channel_idx]
    std = stats['stds'][channel_idx]
    
    # Clip to mean ± zscore*std
    lower_bound = mean - zscore_clip * std
    upper_bound = mean + zscore_clip * std
    magnitude = np.clip(magnitude, lower_bound, upper_bound)
    
    # Standardize: (data - mean) / std
    magnitude = (magnitude - mean) / std
    
    # Ensure float32 for ImageJ compatibility
    magnitude = magnitude.astype(np.float32)
    
    return magnitude

def process_data_mask(data: np.ndarray) -> np.ndarray:
    """Process mask data: extract first channel and convert to float32."""
    data = data[..., 0]
    data = data.astype(np.float32)
    return data

def process_data_mask_dual(data_normal: np.ndarray, data_baseline: np.ndarray) -> np.ndarray:
    """Process dual mask data: stack normal and baseline masks as 2 channels (2, H, W)."""
    # Extract first channel from both masks
    mask_normal = data_normal[..., 0].astype(np.float32)
    mask_baseline = data_baseline[..., 0].astype(np.float32)

    mask_normal[-4:] = 0
    
    # Stack as (2, H, W) where channel 0 = normal, channel 1 = baseline
    mask_stacked = np.stack([mask_normal, mask_baseline], axis=0)
    return mask_stacked

def get_num_channels(input_path: Path) -> int:
    """Get number of channels in a file."""
    data = joblib.load(input_path)
    return data.shape[0]

def process_file_pair(
    img_path: Path,
    mask_path: Path,
    mask_path_baseline: Path,
    output_dir: Path,
    stats: dict,
    zscore_clip: float,
    save_index_offset: int,
) -> None:
    """Process a single image-mask file pair (with normal and baseline masks) and save as paired TIF files."""
    
    try:
        # Load all three files
        img_data = joblib.load(img_path)
        mask_data = joblib.load(mask_path)
        mask_data_baseline = joblib.load(mask_path_baseline)
        
        C = img_data.shape[0]
        
        # Validate matching channel counts
        if mask_data.shape[0] != C:
            logger.warning(f"Channel mismatch: {img_path} has {C} channels, {mask_path} has {mask_data.shape[0]} channels")
            C = min(C, mask_data.shape[0])
        
        if mask_data_baseline.shape[0] != C:
            logger.warning(f"Channel mismatch: {img_path} has {C} channels, {mask_path_baseline} has {mask_data_baseline.shape[0]} channels")
            C = min(C, mask_data_baseline.shape[0])
        
        save_index = save_index_offset
        
        # Process each channel pair
        for i in range(C):
            # Process image
            img_output = process_data_img(
                img_data[i], 
                channel_idx=i,
                stats=stats,
                zscore_clip=zscore_clip,
            )
            
            # Process dual masks (normal + baseline)
            mask_output = process_data_mask_dual(mask_data[i], mask_data_baseline[i])
            
            # Add channel dimension for image TIF
            img_output = img_output[np.newaxis, ...]
            # mask_output already has shape (2, H, W)
            
            # Save paired files with same index
            img_path_out = str(output_dir / f'{save_index}_img.tif')
            mask_path_out = str(output_dir / f'{save_index}_mask.tif')
            
            tif.imwrite(img_path_out, img_output, imagej=True)
            tif.imwrite(mask_path_out, mask_output, imagej=True)
            
            save_index += 1
        
    except Exception as e:
        logger.error(f"Error processing pair {img_path} / {mask_path} / {mask_path_baseline}: {e}")
        raise

def process_pair_wrapper(
    task_tuple: tuple,
    output_dir: Path,
    stats: dict,
    zscore_clip: float,
) -> None:
    """Wrapper to unpack file pair (including baseline mask) and offset for parallel processing."""
    img_path, mask_path, mask_path_baseline, save_index_offset = task_tuple
    process_file_pair(
        img_path=img_path,
        mask_path=mask_path,
        mask_path_baseline=mask_path_baseline,
        output_dir=output_dir,
        stats=stats,
        zscore_clip=zscore_clip,
        save_index_offset=save_index_offset,
    )

def process_pairs_parallel(
    img_paths: list[Path],
    mask_paths: list[Path],
    mask_paths_baseline: list[Path],
    output_dir: Path,
    stats: dict,
    zscore_clip: float,
    save_index_offset: int = 0,
) -> int:
    """
    Process image-mask pairs (with baseline masks) in parallel with pre-computed file index offsets.
    
    Returns:
        int: Next available save index offset after processing all file pairs
    """
    if len(img_paths) == 0:
        logger.warning("No file pairs to process")
        return save_index_offset
    
    # Validate matching counts
    if len(img_paths) != len(mask_paths):
        logger.error(f"Mismatch: {len(img_paths)} image files but {len(mask_paths)} mask files")
        min_count = min(len(img_paths), len(mask_paths))
        img_paths = img_paths[:min_count]
        mask_paths = mask_paths[:min_count]
        mask_paths_baseline = mask_paths_baseline[:min_count]
        logger.warning(f"Processing only first {min_count} pairs")
    
    if len(img_paths) != len(mask_paths_baseline):
        logger.error(f"Mismatch: {len(img_paths)} image files but {len(mask_paths_baseline)} baseline mask files")
        min_count = min(len(img_paths), len(mask_paths_baseline))
        img_paths = img_paths[:min_count]
        mask_paths = mask_paths[:min_count]
        mask_paths_baseline = mask_paths_baseline[:min_count]
        logger.warning(f"Processing only first {min_count} pairs")
    
    logger.info(f"Processing {len(img_paths)} image-mask pairs (with baseline)...")
    
    # Pre-compute channel counts to determine save_index_offset for each file pair
    logger.info("Computing file offsets...")
    channel_counts = [get_num_channels(p) for p in img_paths]
    file_offsets = [save_index_offset + sum(channel_counts[:i]) for i in range(len(channel_counts))]
    
    # Create task tuples (img_path, mask_path, mask_path_baseline, offset) for parallel processing
    tasks = list(zip(img_paths, mask_paths, mask_paths_baseline, file_offsets))
    
    # Process pairs in parallel
    logger.info("Processing pairs in parallel...")
    mapper = ParallelMapper()
    mapper(
        process_pair_wrapper,
        tasks,
        output_dir=output_dir,
        stats=stats,
        zscore_clip=zscore_clip,
    )
    
    # Return the next available save index
    next_offset = save_index_offset + sum(channel_counts)
    logger.info(f"Finished processing {len(img_paths)} pairs (indices {save_index_offset} to {next_offset-1})")
    return next_offset

def main(
    config_path: Path | str | None = None,
) -> None:
    settings = load_settings(config_path, default_settings)
    
    # Setup output directory
    output_dir = setup_directory(
        path=settings['output_dir'], 
        overwrite=settings['overwrite'],
    )
    
    # Load training directories
    train_dirs = load_train_directories(settings['train_dirs_file'])
    
    if len(train_dirs) == 0:
        logger.error("No training directories found. Exiting.")
        return
    
    # Track global save index offset across all directories
    global_save_index = 0
    total_processed = 0
    total_skipped = 0
    
    # Open statistics file for writing
    stats_file = output_dir.parent / 'normalization_statistics.txt'
    with open(stats_file, 'w') as f:
        f.write("# Directory-wise normalization statistics\n")
        f.write("# Format: Directory | Channel | Mean | Std\n\n")
    
    # Process each directory
    for train_dir in train_dirs:
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing directory: {train_dir}")
        logger.info(f"{'='*80}")
        
        # Validate directory
        is_valid, input_dir, label_dir, label_dir_baseline, input_paths, label_paths, label_paths_baseline = validate_directory(train_dir)
        
        if not is_valid:
            logger.warning(f"Skipping invalid directory: {train_dir}")
            total_skipped += 1
            continue
        
        # Collect statistics for this directory's inputs
        stats = collect_image_statistics(input_paths)
        
        # Save statistics to file
        with open(stats_file, 'a') as f:
            f.write(f"Directory: {train_dir}\n")
            for i, (mean, std) in enumerate(zip(stats['means'], stats['stds'])):
                f.write(f"  Channel {i}: mean={mean:.6f}, std={std:.6f}\n")
            f.write("\n")
        
        # Process image-mask pairs with per-directory statistics
        global_save_index = process_pairs_parallel(
            img_paths=input_paths,
            mask_paths=label_paths,
            mask_paths_baseline=label_paths_baseline,
            output_dir=output_dir,
            stats=stats,
            zscore_clip=settings['zscore_clip'],
            save_index_offset=global_save_index,
        )
        
        total_processed += 1
        logger.info(f"Completed directory: {train_dir}")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing complete!")
    logger.info(f"Processed: {total_processed} directories")
    logger.info(f"Skipped: {total_skipped} directories")
    logger.info(f"Total TIF pairs created: {global_save_index}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Statistics saved to: {stats_file}")
    logger.info(f"{'='*80}")

if __name__ == '__main__':
    # python -m autotslabel.autosegment.multichannel.step_6a_convert_tif
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path)