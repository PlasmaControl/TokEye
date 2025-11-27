import sys
from pathlib import Path
import shutil

import numpy as np
import h5py
import tifffile as tif

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .utils.parmap import ParallelMapper
from .utils.configuration import load_settings

default_settings = {
    "input_dir": Path("data/cache/step_6a_convert_tif"),  # Original images and masks
    "predictions_file": Path("data/cache/step_6b_refiner/all_folds_predictions.h5"),
    "output_dir": Path("data/cache/step_6c_refined_masks"),
    "n_folds": 5,
    "mean_threshold": 0.25,
    "entropy_threshold": 0.45,
    "overwrite": True,
}


def load_predictions_for_sample(hdf5_file, sample_idx, n_folds=5):
    """
    Load predictions for a single sample across all folds and average them.

    Args:
        hdf5_file: Open h5py.File object
        sample_idx: Index of the sample
        n_folds: Number of folds to average over

    Returns:
        dict with keys 'mean', 'std', 'entropy', each with shape (2, H, W)
    """
    prediction = {"mean": [], "std": [], "entropy": []}

    for fold_idx in range(n_folds):
        fold_group = hdf5_file[f"fold_{fold_idx}"]

        mean = fold_group["predictions"][sample_idx]
        std = fold_group["std"][sample_idx]
        entropy = fold_group["entropy"][sample_idx]

        prediction["mean"].append(mean)
        prediction["std"].append(std)
        prediction["entropy"].append(entropy)

    # Stack and average across folds
    prediction["mean"] = np.stack(prediction["mean"], axis=0)
    prediction["std"] = np.stack(prediction["std"], axis=0)
    prediction["entropy"] = np.stack(prediction["entropy"], axis=0)

    prediction["mean"] = np.mean(prediction["mean"], axis=0)
    prediction["std"] = np.mean(prediction["std"], axis=0)
    prediction["entropy"] = np.mean(prediction["entropy"], axis=0)

    return prediction


def apply_refinement_logic(
    prediction, input_mask, mean_threshold=0.25, entropy_threshold=0.45
):
    """
    Apply refinement logic to predictions using the original mask.

    Args:
        prediction: dict with 'mean' and 'entropy', each with shape (2, H, W)
        input_mask: Original mask with shape (2, H, W)
        mean_threshold: Threshold for mean predictions
        entropy_threshold: Threshold for entropy

    Returns:
        refined_mask: np.ndarray with shape (2, H, W), dtype float32
    """
    mean = prediction["mean"]
    entropy = prediction["entropy"]

    # Create binary thresholds
    binary_coherent_mean = mean[0] > mean_threshold
    binary_transient_mean = mean[1] > mean_threshold
    binary_coherent_entropy = entropy[0] > entropy_threshold
    binary_transient_entropy = entropy[1] > entropy_threshold

    # Convert input mask to boolean
    input_mask_binary = input_mask.astype(bool)

    # Refine coherent modes (channel 0 / normal)
    refined_coherent_modes = np.logical_and(
        binary_coherent_mean, binary_coherent_entropy
    )
    refined_coherent_modes = np.logical_and(
        refined_coherent_modes,
        np.logical_not(np.logical_or(input_mask_binary[1], binary_transient_mean)),
    )
    refined_coherent_modes = np.logical_or(refined_coherent_modes, input_mask_binary[0])

    # Refine transient modes (channel 1 / baseline)
    refined_transient_modes = np.logical_and(
        binary_transient_mean, binary_transient_entropy
    )
    refined_transient_modes = np.logical_and(
        refined_transient_modes,
        np.logical_not(np.logical_or(input_mask_binary[0], binary_coherent_mean)),
    )
    refined_transient_modes = np.logical_or(
        refined_transient_modes, input_mask_binary[1]
    )

    # Stack refined masks as (2, H, W) and convert to float32
    refined_mask = np.stack(
        [
            refined_coherent_modes.astype(np.float32),
            refined_transient_modes.astype(np.float32),
        ],
        axis=0,
    )

    return refined_mask


def process_single_sample(
    sample_idx: int,
    input_dir: Path,
    output_dir: Path,
    predictions_file: Path,
    n_folds: int,
    mean_threshold: float,
    entropy_threshold: float,
) -> None:
    """
    Process a single sample: load predictions, apply refinement, save refined mask.

    This function is designed to be called in parallel by ParallelMapper.
    Each worker opens its own HDF5 file handle for thread-safe reading.
    """
    # Get file paths
    img_path = input_dir / f"{sample_idx}_img.tif"
    mask_path = input_dir / f"{sample_idx}_mask.tif"

    if not img_path.exists() or not mask_path.exists():
        logger.warning(f"Sample {sample_idx}: Missing input files, skipping")
        return

    # Load original mask
    input_mask = tif.imread(mask_path)

    # Open HDF5 file for this worker (thread-safe)
    with h5py.File(predictions_file, "r") as hdf5_file:
        # Load predictions for this sample
        prediction = load_predictions_for_sample(hdf5_file, sample_idx, n_folds)

    # Apply refinement logic
    refined_mask = apply_refinement_logic(
        prediction,
        input_mask,
        mean_threshold=mean_threshold,
        entropy_threshold=entropy_threshold,
    )

    # Save outputs
    output_img_path = output_dir / f"{sample_idx}_img.tif"
    output_mask_path = output_dir / f"{sample_idx}_mask.tif"

    # Copy input image
    shutil.copy(img_path, output_img_path)

    # Save refined mask
    tif.imwrite(str(output_mask_path), refined_mask, imagej=True)


def process_all_samples(settings):
    """
    Process all samples: load predictions, apply refinement, save refined masks.
    """
    input_dir = settings["input_dir"]
    predictions_file = settings["predictions_file"]
    output_dir = settings["output_dir"]
    n_folds = settings["n_folds"]
    mean_threshold = settings["mean_threshold"]
    entropy_threshold = settings["entropy_threshold"]

    # Create output directory
    if settings.get("overwrite", True) and output_dir.exists():
        logger.info(f"Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all input files
    input_img_files = sorted([p for p in input_dir.glob("*_img.tif")])
    n_samples = len(input_img_files)

    logger.info(f"Found {n_samples} samples to process")
    logger.info(f"Loading predictions from: {predictions_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Thresholds: mean={mean_threshold}, entropy={entropy_threshold}")

    # Create list of sample indices to process
    sample_indices = list(range(n_samples))

    # Process samples in parallel using ParallelMapper
    mapper = ParallelMapper()
    mapper(
        process_single_sample,
        sample_indices,
        input_dir=input_dir,
        output_dir=output_dir,
        predictions_file=predictions_file,
        n_folds=n_folds,
        mean_threshold=mean_threshold,
        entropy_threshold=entropy_threshold,
    )

    logger.info(f"Completed processing {n_samples} samples")
    logger.info(f"Refined masks saved to: {output_dir}")


def main(config_path=None):
    if config_path is None:
        settings = default_settings
    else:
        settings = load_settings(config_path)

    process_all_samples(settings)


if __name__ == "__main__":
    # python -m autotslabel.autosegment.multichannel.step_6c_convert_predictions
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path)
