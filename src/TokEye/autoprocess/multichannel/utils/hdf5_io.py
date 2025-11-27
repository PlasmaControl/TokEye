"""
Utility functions for reading/writing HDF5 files.
Provides helper functions to access predictions from step_6b and step_6c outputs.
"""

from pathlib import Path
import numpy as np
import h5py
from typing import Optional, List, Tuple


def read_fold_prediction(
    hdf5_path: Path, fold_idx: int, sample_idx: int, data_type: str = "pred"
) -> Optional[np.ndarray]:
    """
    Read a single prediction from a fold HDF5 file.

    Args:
        hdf5_path: Path to HDF5 file (e.g., fold_0_predictions.h5)
        fold_idx: Fold index
        sample_idx: Sample index
        data_type: Type of data ('pred', 'std', 'entropy')

    Returns:
        Array of shape (2, H, W) where channel 0 is normal, channel 1 is baseline
        Returns None if not found
    """
    if not hdf5_path.exists():
        return None

    try:
        with h5py.File(hdf5_path, "r") as f:
            fold_group = f[f"fold_{fold_idx}"]
            dataset_name = f"sample_{sample_idx}_{data_type}"
            if dataset_name in fold_group:
                return fold_group[dataset_name][:]
            else:
                return None
    except (KeyError, OSError):
        return None


def read_ensemble_prediction(
    hdf5_path: Path, sample_idx: int, data_type: str = "ensemble_mean"
) -> Optional[np.ndarray]:
    """
    Read an ensemble prediction from the ensemble HDF5 file.

    Args:
        hdf5_path: Path to ensemble HDF5 file (e.g., ensemble.h5)
        sample_idx: Sample index
        data_type: Type of data ('ensemble_mean', 'ensemble_std', 'ensemble_entropy',
                                  'mc_std_mean', 'mc_entropy_mean')

    Returns:
        Array of shape (2, H, W) where channel 0 is normal, channel 1 is baseline
        Returns None if not found
    """
    if not hdf5_path.exists():
        return None

    try:
        with h5py.File(hdf5_path, "r") as f:
            dataset_name = f"sample_{sample_idx}_{data_type}"
            if dataset_name in f:
                return f[dataset_name][:]
            else:
                return None
    except (KeyError, OSError):
        return None


def get_channel(data: np.ndarray, channel: str = "normal") -> np.ndarray:
    """
    Extract a specific channel from 2-channel prediction data.

    Args:
        data: Array of shape (2, H, W)
        channel: 'normal' (channel 0) or 'baseline' (channel 1)

    Returns:
        Array of shape (H, W) for the specified channel
    """
    channel_idx = 0 if channel == "normal" else 1
    return data[channel_idx]


def list_samples(hdf5_path: Path, fold_idx: Optional[int] = None) -> List[int]:
    """
    List all sample indices in an HDF5 file.

    Args:
        hdf5_path: Path to HDF5 file
        fold_idx: Fold index (for step_6b output), or None for step_6c output

    Returns:
        Sorted list of sample indices
    """
    if not hdf5_path.exists():
        return []

    try:
        with h5py.File(hdf5_path, "r") as f:
            if fold_idx is not None:
                # Step 6b output
                fold_group = f[f"fold_{fold_idx}"]
                keys = fold_group.keys()
            else:
                # Step 6c output
                keys = f.keys()

            # Extract sample indices from dataset names
            sample_indices = set()
            for key in keys:
                if "sample_" in key:
                    idx_str = key.split("_")[1]
                    try:
                        sample_indices.add(int(idx_str))
                    except ValueError:
                        pass

            return sorted(list(sample_indices))
    except (KeyError, OSError):
        return []


def get_metadata(hdf5_path: Path, fold_idx: Optional[int] = None) -> dict:
    """
    Get metadata from an HDF5 file.

    Args:
        hdf5_path: Path to HDF5 file
        fold_idx: Fold index (for step_6b output), or None for step_6c output

    Returns:
        Dictionary of metadata attributes
    """
    if not hdf5_path.exists():
        return {}

    try:
        with h5py.File(hdf5_path, "r") as f:
            if fold_idx is not None:
                # Step 6b output
                fold_group = f[f"fold_{fold_idx}"]
                return dict(fold_group.attrs)
            else:
                # Step 6c output
                return dict(f.attrs)
    except (KeyError, OSError):
        return {}


def read_all_samples(
    hdf5_path: Path, fold_idx: Optional[int] = None, data_type: str = "pred"
) -> Tuple[List[int], np.ndarray]:
    """
    Read all samples from an HDF5 file.

    Args:
        hdf5_path: Path to HDF5 file
        fold_idx: Fold index (for step_6b output), or None for step_6c output
        data_type: Type of data to load

    Returns:
        Tuple of (sample_indices, data_array)
        data_array has shape (N, 2, H, W)
    """
    sample_indices = list_samples(hdf5_path, fold_idx)

    if not sample_indices:
        return [], np.array([])

    # Load all samples
    samples = []
    valid_indices = []

    for idx in sample_indices:
        if fold_idx is not None:
            # Step 6b output
            data = read_fold_prediction(hdf5_path, fold_idx, idx, data_type)
        else:
            # Step 6c output
            data = read_ensemble_prediction(hdf5_path, idx, data_type)

        if data is not None:
            samples.append(data)
            valid_indices.append(idx)

    if samples:
        return valid_indices, np.stack(samples, axis=0)
    else:
        return [], np.array([])


# Example usage functions
def example_read_fold_predictions():
    """Example: Read predictions from step_6b output."""
    from pathlib import Path

    hdf5_path = Path("data/cache/step_6b_segmenter/fold_0_predictions.h5")
    fold_idx = 0
    sample_idx = 0

    # Read prediction for a specific sample
    pred = read_fold_prediction(hdf5_path, fold_idx, sample_idx, "pred")

    if pred is not None:
        print(f"Prediction shape: {pred.shape}")  # (2, H, W)

        # Extract individual channels
        normal = get_channel(pred, "normal")  # (H, W)
        baseline = get_channel(pred, "baseline")  # (H, W)

        print(f"Normal channel shape: {normal.shape}")
        print(f"Baseline channel shape: {baseline.shape}")


def example_read_ensemble():
    """Example: Read ensemble predictions from step_6c output."""
    from pathlib import Path

    hdf5_path = Path("data/cache/step_6c_ensemble/ensemble.h5")
    sample_idx = 0

    # Read ensemble mean
    ensemble_mean = read_ensemble_prediction(hdf5_path, sample_idx, "ensemble_mean")

    if ensemble_mean is not None:
        print(f"Ensemble mean shape: {ensemble_mean.shape}")  # (2, H, W)

        # Extract individual channels
        normal_mean = get_channel(ensemble_mean, "normal")
        baseline_mean = get_channel(ensemble_mean, "baseline")

        print(f"Normal channel mean shape: {normal_mean.shape}")
        print(f"Baseline channel mean shape: {baseline_mean.shape}")
