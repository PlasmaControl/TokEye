import json
import logging
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import tifffile as tif
from scipy.ndimage import zoom
from tqdm.auto import tqdm

from .configuration import load_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_DIR = Path("/scratch/gpfs/nc1514/autotslabel/data/cache")
STEP_9_DIR = CACHE_DIR / "step_9_remove_sparse_data"
STEP_10_DIR = CACHE_DIR / "step_10_to_instance"

default_settings = {
    "nnunet": {
        "step_9_dir": STEP_9_DIR,
        "output_base_dir": CACHE_DIR / "nnunet" / "raw",
        "dataset_name": "Dataset002_ECEBinary",
        "overwrite": True,
    },
    "omnipose": {
        "step_9_dir": STEP_9_DIR,
        "step_10_dir": STEP_10_DIR,
        "output_base_dir": CACHE_DIR / "omnipose",
        "dataset_name": "default_dataset",
        "overwrite": True,
    },
}


def setup_dir(path, overwrite=True):
    """Setup output directory."""
    path = Path(path)
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_image_files(step_dir):
    """Get sorted list of non-mask image files."""
    return sorted(
        [f for f in Path(step_dir).glob("*.tif") if not f.stem.endswith("_masks")],
        key=lambda x: int(x.stem),
    )


def _process_nnunet_file(args):
    """Process a single file for nnUNet."""
    img_file, step_9_dir, images_tr_dir, labels_tr_dir = args
    idx = img_file.stem
    mask_file = step_9_dir / f"{idx}_masks.tif"
    if not mask_file.exists():
        return False, f"Mask not found for {idx}"
    try:
        tif.imwrite(images_tr_dir / f"{idx}_0000.tif", tif.imread(img_file))
        tif.imwrite(labels_tr_dir / f"{idx}.tif", tif.imread(mask_file))
        return True, idx
    except Exception as e:
        return False, f"Error {idx}: {e}"


def _process_omnipose_file(args):
    """Process a single file for Omnipose."""
    img_file, step_9_dir, step_10_dir, dataset_dir = args
    idx = img_file.stem
    mask_file = step_10_dir / f"{idx}_masks.tif"
    if not mask_file.exists():
        return False, f"Mask not found for {idx}"
    try:
        img = tif.imread(img_file)
        mask = tif.imread(mask_file)

        # Calculate zoom factors
        target_size = (512, 512)
        zoom_factors = (target_size[0] / img.shape[0], target_size[1] / img.shape[1])
        img = zoom(img, zoom_factors, order=1)
        mask = zoom(mask, zoom_factors, order=0)
        img = np.expand_dims(img, axis=0)
        tif.imwrite(dataset_dir / f"{idx}_img.tif", img)
        tif.imwrite(dataset_dir / f"{idx}_masks.tif", mask)
        return True, idx
    except Exception as e:
        return False, f"Error {idx}: {e}"


def convert_nnunet(
    step_9_dir=None,
    output_base_dir=None,
    dataset_name="Dataset001_Binary",
    overwrite=True,
    max_workers=None,
):
    """Convert step 9 outputs to nnUNet format."""
    step_9_dir = Path(step_9_dir or STEP_9_DIR)
    dataset_dir = setup_dir(
        Path(output_base_dir or CACHE_DIR / "nnunet") / dataset_name, overwrite
    )
    images_tr_dir = setup_dir(dataset_dir / "imagesTr", overwrite=False)
    labels_tr_dir = setup_dir(dataset_dir / "labelsTr", overwrite=False)

    image_files = get_image_files(step_9_dir)
    logger.info(f"Converting {len(image_files)} files to nnUNet format")

    args_list = [(f, step_9_dir, images_tr_dir, labels_tr_dir) for f in image_files]
    workers = max_workers or min(cpu_count(), len(image_files))

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_process_nnunet_file, args) for args in args_list]
        with tqdm(total=len(image_files), desc="nnUNet") as pbar:
            for future in as_completed(futures):
                success, msg = future.result()
                if not success:
                    logger.warning(msg)
                pbar.update(1)

    num_training = len(list(images_tr_dir.glob("*.tif")))
    with (dataset_dir / "dataset.json").open("w") as f:
        json.dump(
            {
                "channel_names": {"0": "Channel1"},
                "labels": {"background": 0, "foreground": 1},
                "numTraining": num_training,
                "file_ending": ".tif",
            },
            f,
            indent=2,
        )

    logger.info(f"Converted {num_training} pairs")
    return dataset_dir


def convert_omnipose(
    step_9_dir=None,
    step_10_dir=None,
    output_base_dir=None,
    dataset_name="default_dataset",
    overwrite=True,
    max_workers=None,
):
    """Convert step 9 images + step 10 instance masks to Omnipose format."""
    step_9_dir = Path(step_9_dir or STEP_9_DIR)
    step_10_dir = Path(step_10_dir or STEP_10_DIR)
    dataset_dir = setup_dir(
        Path(output_base_dir or CACHE_DIR / "omnipose") / dataset_name, overwrite
    )

    image_files = get_image_files(step_9_dir)
    logger.info(f"Converting {len(image_files)} files to Omnipose format")

    args_list = [(f, step_9_dir, step_10_dir, dataset_dir) for f in image_files]
    workers = max_workers or min(cpu_count(), len(image_files))

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_process_omnipose_file, args) for args in args_list]
        with tqdm(total=len(image_files), desc="Omnipose") as pbar:
            for future in as_completed(futures):
                success, msg = future.result()
                if not success:
                    logger.warning(msg)
                pbar.update(1)

    converted = len(list(dataset_dir.glob("*_masks.tif")))
    logger.info(f"Converted {converted} pairs")
    return dataset_dir


def main(config_path=None):
    """Convert data to nnUNet and/or Omnipose formats."""
    settings = default_settings if config_path is None else load_settings(config_path)

    if nnunet := settings.get("nnunet"):
        logger.info("Converting to nnUNet format...")
        convert_nnunet(**nnunet)

    # if omnipose := settings.get('omnipose', {'step_9_dir': STEP_9_DIR, 'step_10_dir': STEP_10_DIR}):
    #     logger.info("Converting to Omnipose format...")
    #     convert_omnipose(**omnipose)


if __name__ == "__main__":
    # python -m autotslabel.autosegment.multichannel.utils.convert_for_software
    main(sys.argv[1] if len(sys.argv) > 1 else None)
