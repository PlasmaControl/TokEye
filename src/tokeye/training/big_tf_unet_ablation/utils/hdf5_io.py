"""
Utility functions for reading/writing HDF5 files.

Provides:
- Generic step-level I/O (create, write, read, iterate) for the multiscale
  pipeline's intermediate HDF5 files.
- ``HDF5StepDataset`` — a PyTorch Dataset backed by an HDF5 step file.
- Original step_6b/6c prediction helpers (unchanged).
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from collections.abc import Iterator

# =====================================================================
# Generic step-level HDF5 I/O
# =====================================================================


def create_step_file(
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> h5py.File:
    """Create a new HDF5 step file with a ``/samples`` group and metadata.

    Returns the open file handle — caller must close it (or use as context
    manager).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = h5py.File(path, "w")
    f.create_group("samples")
    if metadata:
        for k, v in metadata.items():
            if v is not None:
                f.attrs[k] = v
    return f


def write_sample(
    h5_file: h5py.File,
    idx: int,
    data: np.ndarray,
    group: str = "samples",
) -> None:
    """Write one sample to an HDF5 file with LZF compression."""
    grp = h5_file.require_group(group)
    ds_name = str(idx)
    if ds_name in grp:
        del grp[ds_name]
    grp.create_dataset(ds_name, data=data, compression="lzf")


def read_sample(
    h5_file: h5py.File | str | Path,
    idx: int,
    group: str = "samples",
) -> np.ndarray:
    """Read one sample from an HDF5 file."""
    if isinstance(h5_file, (str, Path)):
        with h5py.File(h5_file, "r") as f:
            return np.asarray(f[group][str(idx)])
    return np.asarray(h5_file[group][str(idx)])


def get_sample_count(h5_path: str | Path, group: str = "samples") -> int:
    """Return the number of samples in an HDF5 step file."""
    with h5py.File(h5_path, "r") as f:
        if group not in f:
            return 0
        return len(f[group])


def iter_samples(
    h5_path: str | Path,
    group: str = "samples",
) -> Iterator[tuple[int, np.ndarray]]:
    """Lazily iterate ``(idx, array)`` pairs from an HDF5 step file."""
    with h5py.File(h5_path, "r") as f:
        grp = f[group]
        for key in sorted(grp.keys(), key=int):
            yield int(key), np.asarray(grp[key])


def get_step_metadata(h5_path: str | Path) -> dict[str, Any]:
    """Read top-level metadata attrs from an HDF5 step file."""
    with h5py.File(h5_path, "r") as f:
        return dict(f.attrs)


# =====================================================================
# PyTorch Dataset backed by an HDF5 step file
# =====================================================================


class HDF5StepDataset(Dataset):
    """PyTorch Dataset that reads samples from an HDF5 step file.

    Opens the file lazily on first access (or in ``worker_init``) and
    reads one sample per ``__getitem__`` call — no full-file load.
    """

    def __init__(
        self,
        h5_path: str | Path,
        group: str = "samples",
        transform=None,
    ) -> None:
        self.h5_path = str(h5_path)
        self.group = group
        self.transform = transform

        # Determine length and sorted keys up-front
        with h5py.File(self.h5_path, "r") as f:
            grp = f[self.group]
            self._keys = sorted(grp.keys(), key=int)
        self._h5: h5py.File | None = None

    def _open(self) -> None:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", swmr=True)

    def worker_init(self) -> None:
        """Call from DataLoader ``worker_init_fn`` to open per-worker handle."""
        self._open()

    def __len__(self) -> int:
        return len(self._keys)

    def __getitem__(self, idx: int) -> torch.Tensor | Any:
        self._open()
        data = np.asarray(self._h5[self.group][self._keys[idx]])
        data = torch.from_numpy(data).float()
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __del__(self) -> None:
        if self._h5 is not None:
            self._h5.close()


def worker_init_fn(worker_id: int) -> None:
    """Generic ``worker_init_fn`` for DataLoaders using HDF5StepDataset."""
    import torch.utils.data as data_utils

    dataset = data_utils.get_worker_info().dataset
    if hasattr(dataset, "worker_init"):
        dataset.worker_init()


# =====================================================================
# Original step_6b/6c prediction helpers (unchanged API)
# =====================================================================


def read_fold_prediction(
    hdf5_path: Path, fold_idx: int, sample_idx: int, data_type: str = "pred"
) -> np.ndarray | None:
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
            return None
    except (KeyError, OSError):
        return None


def read_ensemble_prediction(
    hdf5_path: Path, sample_idx: int, data_type: str = "ensemble_mean"
) -> np.ndarray | None:
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


def list_samples(hdf5_path: Path, fold_idx: int | None = None) -> list[int]:
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
                    with contextlib.suppress(ValueError):
                        sample_indices.add(int(idx_str))

            return sorted(sample_indices)
    except (KeyError, OSError):
        return []


def get_metadata(hdf5_path: Path, fold_idx: int | None = None) -> dict:
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
            # Step 6c output
            return dict(f.attrs)
    except (KeyError, OSError):
        return {}


def read_all_samples(
    hdf5_path: Path, fold_idx: int | None = None, data_type: str = "pred"
) -> tuple[list[int], np.ndarray]:
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
