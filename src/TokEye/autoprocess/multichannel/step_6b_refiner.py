import sys

from pathlib import Path
import shutil
import json

import tifffile as tif
import numpy as np
import h5py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

torch.backends.cuda.matmul.fp32_precision = "ieee"
device = "cuda" if torch.cuda.is_available() else "cpu"

import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    BasePredictionWriter,
)

L.seed_everything(42)

from sklearn.model_selection import KFold

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .utils.configuration import load_settings
from .utils.augmentations import get_augmentation
from .utils.losses import get_loss_function, dice_coefficient, iou_score

from TokEye.models.unet import UNet

default_settings = {
    # Data and training
    "batch_size": 56,
    "num_workers": 6,
    "prefetch_factor": 2,
    "num_layers": 5,
    "first_layer_size": 32,
    "max_epochs": 200,
    "precision": "bf16-mixed",
    "devices": 1,
    "enable_progress_bar": False,
    "ckpt_path": None,
    "log_every_n_steps": 10,
    "input_dir": Path("data/cache/step_6a_convert_tif"),
    "output_dir": Path("data/cache/step_6b_refiner"),
    "model_dir": Path("models/segmenter"),
    "fast_dev_run": False,
    "early_stopping_patience": 3,
    "n_folds": 5,
    # Augmentation settings
    "augmentation": True,
    "aug_rotation_degrees": 180,
    "aug_prob_flip": 0.5,
    "aug_elastic": True,
    "aug_elastic_alpha": 50.0,
    "aug_elastic_sigma": 5.0,
    "aug_scale_range": [0.8, 1.2],
    "aug_intensity": True,
    "aug_brightness_range": [0.8, 1.2],
    "aug_contrast_range": [0.8, 1.2],
    "aug_noise_std": 0.05,
    "aug_blur_prob": 0.3,
    "aug_blur_sigma_range": [0.5, 1.5],
    "aug_gamma_range": [0.8, 1.2],
    "aug_apply_prob": 0.8,
    # SpecAugment settings (for spectrogram-specific augmentation)
    "specaugment": True,
    "specaug_time_warp_W": 20,  # time warp parameter W (max displacement)
    "specaug_freq_mask_F": 5,  # max frequency mask width
    "specaug_time_mask_T": 5,  # max time mask width
    "specaug_freq_mask_num": 0,  # number of frequency masks (0=disabled by default)
    "specaug_time_mask_num": 0,  # number of time masks (0=disabled by default)
    # Loss function settings
    "loss_type": "symmetric_bce_dice",  # Options: 'bce', 'label_smooth_bce', 'symmetric_bce', 'dice', 'dice_bce', 'symmetric_bce_dice', 'focal', 'focal_dice', 'iou'
    "label_smoothing": 0.1,
    "symmetric_alpha": 0.1,
    "symmetric_beta": 1.0,
    "symmetric_weight": 0.5,
    "dice_weight": 0.5,
    "bce_weight": 0.5,
    "focal_alpha": 0.25,
    "focal_gamma": 2.0,
    "focal_weight": 0.5,
    # Optimizer settings
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    # Learning rate scheduler
    "lr_scheduler": "reduce_on_plateau",  # Options: None, 'reduce_on_plateau', 'cosine'
    "lr_factor": 0.5,
    "lr_patience": 5,
    "lr_min": 1e-6,
    # MC Dropout and uncertainty
    "dropout_rate": 0.2,
    "mc_dropout_samples": 15,
    "save_uncertainty": True,
}


class TiffDataset(Dataset):
    def __init__(self, data_files, transform=None, settings=None):
        self.data_files = data_files
        self.transform = transform
        self.settings = settings if settings is not None else {}

    def _get_label_path(self, input_path):
        return input_path.with_name(input_path.name.replace("_img.tif", "_mask.tif"))

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        input_path = self.data_files[idx]
        label_path = self._get_label_path(input_path)
        x, y = tif.imread(input_path), tif.imread(label_path)

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        if len(y.shape) == 2:
            y = y.unsqueeze(0)

        # Apply augmentation if provided
        if self.transform is not None:
            x, y = self.transform(x, y)

        return x, y


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size=4,
        num_workers=4,
        prefetch_factor=4,
        train_indices=None,
        val_indices=None,
        settings=None,
    ):
        super().__init__()
        self.data_dir = data_dir

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.settings = settings if settings is not None else {}

    def setup(self, stage=None):
        # Get and sort data files once
        self.data_files = [
            p for p in self.data_dir.glob("*.tif") if "_mask" not in p.name
        ]
        self.data_files.sort(key=lambda x: int(x.stem.replace("_img", "")))

        # Create base datasets with appropriate transforms
        train_augmentation = get_augmentation(self.settings)
        full_train_dataset = TiffDataset(
            data_files=self.data_files,
            transform=train_augmentation,
            settings=self.settings,
        )
        full_val_dataset = TiffDataset(
            data_files=self.data_files, transform=None, settings=self.settings
        )

        # Create subsets for train/val splits
        self.train_dataset = Subset(full_train_dataset, self.train_indices)
        self.val_dataset = Subset(full_val_dataset, self.val_indices)

        # Prediction uses full dataset without augmentation (same as val)
        self.pred_dataset = full_val_dataset

    def _create_dataloader(self, dataset, shuffle=False, persistent_workers=False):
        """Helper to create DataLoader with common settings."""
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=persistent_workers,
            prefetch_factor=self.prefetch_factor if persistent_workers else None,
        )

    def train_dataloader(self):
        return self._create_dataloader(
            self.train_dataset, shuffle=True, persistent_workers=True
        )

    def val_dataloader(self):
        return self._create_dataloader(
            self.val_dataset, shuffle=False, persistent_workers=False
        )

    def predict_dataloader(self):
        return self._create_dataloader(
            self.pred_dataset, shuffle=False, persistent_workers=False
        )


class Module(L.LightningModule):
    def __init__(
        self,
        num_layers=4,
        first_layer_size=16,
        settings=default_settings,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.first_layer_size = first_layer_size
        self.settings = settings

        self.unet = UNet(
            in_channels=1,
            out_channels=2,  # 2 channels: normal (ch0) and baseline (ch1)
            num_layers=num_layers,
            first_layer_size=first_layer_size,
            dropout_rate=settings.get("dropout_rate", 0.0),
        )
        if not self.settings["fast_dev_run"]:
            self.unet.compile()

        # Use noise-robust loss function
        self.loss_fn = get_loss_function(settings)

    def forward(self, x):
        return self.unet(x)

    def _shared_step(self, batch, stage):
        """
        Shared logic for training and validation steps.

        Args:
            batch: Input batch (x, y)
            stage: 'train' or 'val'

        Returns:
            Total loss
        """
        x, y = batch
        y_hat = self(x)

        # Channel names for logging
        channel_names = ["normal", "baseline"]

        # Compute loss and metrics per channel
        losses = []
        dice_scores = []
        iou_scores = []

        for ch_idx, ch_name in enumerate(channel_names):
            # Extract channel predictions and targets
            y_hat_ch = y_hat[:, ch_idx : ch_idx + 1, ...]
            y_ch = y[:, ch_idx : ch_idx + 1, ...]

            # Compute metrics
            loss_ch = self.loss_fn(y_hat_ch, y_ch)
            dice_ch = dice_coefficient(y_hat_ch, y_ch)
            iou_ch = iou_score(y_hat_ch, y_ch)

            losses.append(loss_ch)
            dice_scores.append(dice_ch)
            iou_scores.append(iou_ch)

            # Log per-channel metrics
            self.log(f"{stage}_loss_{ch_name}", loss_ch)
            self.log(f"{stage}_dice_{ch_name}", dice_ch)
            self.log(f"{stage}_iou_{ch_name}", iou_ch)

        # Compute average metrics
        total_loss = sum(losses) / len(losses)
        avg_dice = sum(dice_scores) / len(dice_scores)
        avg_iou = sum(iou_scores) / len(iou_scores)

        # Log average metrics
        self.log(f"{stage}_loss", total_loss, prog_bar=True)
        self.log(f"{stage}_dice", avg_dice, prog_bar=True)
        self.log(f"{stage}_iou", avg_iou)

        return total_loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def predict_step(self, batch, batch_idx):
        with torch.inference_mode():
            x, y = batch

            # MC Dropout: multiple forward passes with dropout enabled
            mc_samples = self.settings.get("mc_dropout_samples", 1)
            save_uncertainty = self.settings.get("save_uncertainty", False)

            if mc_samples > 1 and save_uncertainty:
                # Enable dropout for MC sampling
                self.unet.train()  # Enable dropout

                predictions = []
                for _ in range(mc_samples):
                    y_hat_sample = self(x)
                    y_hat_sample = y_hat_sample.sigmoid()
                    predictions.append(y_hat_sample)

                # Stack predictions: (mc_samples, B, C, H, W)
                predictions = torch.stack(predictions, dim=0)

                # Compute mean and std (epistemic uncertainty)
                y_hat_mean = predictions.mean(dim=0)
                y_hat_std = predictions.std(dim=0)

                # Compute entropy (total uncertainty)
                # H = -p*log(p) - (1-p)*log(1-p)
                eps = 1e-7
                y_hat_mean_clipped = torch.clamp(y_hat_mean, eps, 1 - eps)
                entropy = -(
                    y_hat_mean_clipped * torch.log(y_hat_mean_clipped)
                    + (1 - y_hat_mean_clipped) * torch.log(1 - y_hat_mean_clipped)
                )

                self.unet.eval()  # Disable dropout after MC sampling

            else:
                # Standard prediction without MC Dropout
                y_hat_mean = self(x).sigmoid()
                y_hat_std = None
                entropy = None

            # Return predictions as dictionary for HDF5 storage
            # Format: (B, 2, H, W) - batch, channels, height, width
            results = {
                "pred": y_hat_mean.float().cpu().numpy(),  # (B, 2, H, W)
            }

            if save_uncertainty and y_hat_std is not None and entropy is not None:
                results["std"] = y_hat_std.float().cpu().numpy()  # (B, 2, H, W)
                results["entropy"] = entropy.float().cpu().numpy()  # (B, 2, H, W)

            return results

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.settings.get("learning_rate", 1e-4),
            weight_decay=self.settings.get("weight_decay", 1e-5),
        )

        lr_scheduler_type = self.settings.get("lr_scheduler", None)

        if lr_scheduler_type == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.settings.get("lr_factor", 0.5),
                patience=self.settings.get("lr_patience", 5),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        elif lr_scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.settings.get("max_epochs", 40),
                eta_min=self.settings.get("lr_min", 1e-6),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        else:
            return optimizer


class HDF5PredictionWriter(BasePredictionWriter):
    """
    Custom callback to write predictions directly to HDF5 file as they are generated.
    This prevents OOM issues by not accumulating predictions in memory.
    Keeps the HDF5 file open across batches for better I/O performance.
    Uses preallocated arrays for faster batch writes with LZF compression.
    """

    def __init__(
        self,
        hdf5_path,
        fold_idx,
        n_samples,
        save_uncertainty=False,
        write_interval="batch",
    ):
        super().__init__(write_interval)
        self.hdf5_path = hdf5_path
        self.fold_idx = fold_idx
        self.n_samples = n_samples
        self.save_uncertainty = save_uncertainty
        self.batch_idx = 0
        self.total_samples = 0
        self.hdf5_file = None
        self.fold_group = None
        # Datasets for batch writing
        self.pred_dataset = None
        self.std_dataset = None
        self.entropy_dataset = None

    def on_predict_start(self, trainer, pl_module):
        """Called when prediction starts. Opens HDF5 file and creates fold group."""
        try:
            self.hdf5_file = h5py.File(self.hdf5_path, "a")

            # Create fold group if it doesn't exist
            fold_group_name = f"fold_{self.fold_idx}"
            if fold_group_name not in self.hdf5_file:
                self.fold_group = self.hdf5_file.create_group(fold_group_name)
                self.fold_group.attrs["fold_idx"] = self.fold_idx
                self.fold_group.attrs["channels"] = "0: normal, 1: baseline"
            else:
                self.fold_group = self.hdf5_file[fold_group_name]

            logger.info(
                f"Fold {self.fold_idx}: Opened HDF5 file for batch prediction writing"
            )
        except Exception as e:
            logger.error(f"Fold {self.fold_idx}: Failed to open HDF5 file: {e}")
            if self.hdf5_file is not None:
                self.hdf5_file.close()
            raise

    def _initialize_datasets(self, sample_shape):
        """
        Initialize preallocated HDF5 datasets based on first batch shape.

        Args:
            sample_shape: Shape of a single sample (n_channels, H, W)
        """
        n_channels, height, width = sample_shape

        # Create prediction dataset with LZF compression
        self.pred_dataset = self.fold_group.create_dataset(
            "predictions",
            shape=(self.n_samples, n_channels, height, width),
            dtype=np.float32,
            compression="lzf",
            chunks=(1, n_channels, height, width),
        )

        # Create uncertainty datasets if needed
        if self.save_uncertainty:
            self.std_dataset = self.fold_group.create_dataset(
                "std",
                shape=(self.n_samples, n_channels, height, width),
                dtype=np.float32,
                compression="lzf",
                chunks=(1, n_channels, height, width),
            )
            self.entropy_dataset = self.fold_group.create_dataset(
                "entropy",
                shape=(self.n_samples, n_channels, height, width),
                dtype=np.float32,
                compression="lzf",
                chunks=(1, n_channels, height, width),
            )

        # Store metadata
        self.fold_group.attrs["n_samples"] = self.n_samples
        self.fold_group.attrs["shape"] = str(sample_shape)

        logger.info(
            f"Fold {self.fold_idx}: Initialized preallocated datasets with shape {(self.n_samples,) + sample_shape}"
        )

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        """
        Called at the end of each prediction batch.
        Writes entire batch directly to preallocated HDF5 arrays.
        """
        # Free up GPU memory after prediction
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Ensure fold group is initialized
        if self.fold_group is None:
            raise RuntimeError(
                "HDF5 fold group not initialized. on_predict_start may have failed."
            )

        # Get predictions from batch (already numpy arrays from predict_step)
        preds = prediction["pred"]  # (B, 2, H, W)
        batch_size = preds.shape[0]

        # Initialize datasets on first batch when we know the shape
        if self.pred_dataset is None:
            sample_shape = preds[0].shape  # (2, H, W)
            self._initialize_datasets(sample_shape)

        # Calculate slice indices for this batch
        start_idx = self.total_samples
        end_idx = start_idx + batch_size

        # Ensure datasets are initialized
        if self.pred_dataset is None:
            raise RuntimeError("Prediction dataset not initialized")

        # Write entire batch at once using array slicing
        self.pred_dataset[start_idx:end_idx] = preds

        # Write uncertainty maps if available
        if "std" in prediction and self.std_dataset is not None:
            self.std_dataset[start_idx:end_idx] = prediction["std"]

        if "entropy" in prediction and self.entropy_dataset is not None:
            self.entropy_dataset[start_idx:end_idx] = prediction["entropy"]

        self.total_samples += batch_size
        self.batch_idx += 1

        if self.batch_idx % 10 == 0:
            logger.info(
                f"Fold {self.fold_idx}: Saved {self.total_samples}/{self.n_samples} predictions..."
            )

    def on_predict_end(self, trainer, pl_module):
        """Called when prediction ends. Closes HDF5 file."""
        try:
            if self.hdf5_file is not None:
                self.hdf5_file.close()
                logger.info(
                    f"Fold {self.fold_idx}: Successfully saved {self.total_samples} total predictions and closed file"
                )
        except Exception as e:
            logger.error(f"Fold {self.fold_idx}: Error closing HDF5 file: {e}")
            raise


def main(config_path=None):
    if config_path is None:
        settings = default_settings
    else:
        settings = load_settings(config_path)

    input_dir = settings["input_dir"]
    output_dir = settings["output_dir"]
    model_dir = settings["model_dir"]
    n_folds = settings["n_folds"]

    # Clean up and create directories
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Get all data files
    data_files = [p for p in input_dir.glob("*.tif") if "_mask" not in p.name]
    data_files.sort(key=lambda x: int(x.stem.replace("_img", "")))
    n_samples = len(data_files)

    logger.info(f"Found {n_samples} data files")
    logger.info(f"Performing {n_folds}-fold cross-validation")

    # Create K-Fold splits
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    indices = np.arange(n_samples)

    # Save fold indices to file
    fold_info = {
        "n_samples": n_samples,
        "n_folds": n_folds,
        "data_files": [str(f) for f in data_files],
        "folds": [],
    }

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices)):
        fold_info["folds"].append(
            {
                "fold": fold_idx,
                "train_indices": train_idx.tolist(),
                "val_indices": val_idx.tolist(),
            }
        )

    fold_indices_path = output_dir / "fold_indices.json"
    with open(fold_indices_path, "w") as f:
        json.dump(fold_info, f, indent=2)
    logger.info(f"Saved fold indices to {fold_indices_path}")

    # Create single HDF5 file for all fold predictions with initial metadata
    hdf5_path = output_dir / "all_folds_predictions.h5"
    with h5py.File(hdf5_path, "w") as hdf5_file:
        # Add global metadata
        hdf5_file.attrs["n_folds"] = n_folds
        hdf5_file.attrs["n_samples"] = n_samples
        hdf5_file.attrs["channels"] = "0: normal, 1: baseline"
    logger.info(f"Created HDF5 file: {hdf5_path}")

    # Train model for each fold
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices)):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Training Fold {fold_idx + 1}/{n_folds}")
        logger.info(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
        logger.info(f"{'=' * 60}\n")

        # Create fold-specific directories
        fold_model_dir = model_dir / f"fold_{fold_idx}"
        fold_model_dir.mkdir(parents=True, exist_ok=True)

        # Create data module with fold-specific indices
        datamodule = DataModule(
            data_dir=input_dir,
            batch_size=settings["batch_size"],
            num_workers=settings["num_workers"],
            prefetch_factor=settings["prefetch_factor"],
            train_indices=train_idx.tolist(),
            val_indices=val_idx.tolist(),
            settings=settings,
        )

        # Create model
        fold_settings = settings.copy()

        model = Module(
            num_layers=settings["num_layers"],
            first_layer_size=settings["first_layer_size"],
            settings=fold_settings,
        )

        # Set up callbacks for training
        checkpoint_callback = ModelCheckpoint(
            dirpath=fold_model_dir,
            filename="best_model",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )

        early_stopping_callback = EarlyStopping(
            monitor="val_loss",
            patience=settings["early_stopping_patience"],
            mode="min",
        )

        train_callbacks = [checkpoint_callback, early_stopping_callback]

        # Create trainer for training
        trainer = L.Trainer(
            max_epochs=settings["max_epochs"],
            precision=settings["precision"],
            devices=settings["devices"],
            log_every_n_steps=settings["log_every_n_steps"],
            enable_progress_bar=settings["enable_progress_bar"],
            fast_dev_run=settings["fast_dev_run"],
            callbacks=train_callbacks,
        )

        # Train
        trainer.fit(model, datamodule)

        # Load best model for prediction
        best_model_path = fold_model_dir / "best_model.ckpt"
        if best_model_path.exists():
            logger.info(f"Loading best model from {best_model_path}")
            model = Module.load_from_checkpoint(
                best_model_path,
                num_layers=settings["num_layers"],
                first_layer_size=settings["first_layer_size"],
                settings=fold_settings,
            )

        # Set up callback for prediction that saves directly to HDF5
        prediction_writer = HDF5PredictionWriter(
            hdf5_path=hdf5_path,
            fold_idx=fold_idx,
            n_samples=n_samples,
            save_uncertainty=settings.get("save_uncertainty", False),
        )

        # Create new trainer for prediction with the writer callback
        predict_trainer = L.Trainer(
            precision=settings["precision"],
            devices=settings["devices"],
            enable_progress_bar=settings["enable_progress_bar"],
            callbacks=[prediction_writer],
        )

        # Run predictions on all data (return_predictions=False prevents accumulation in memory)
        logger.info(
            f"Running predictions for fold {fold_idx} (batch writing to preallocated HDF5 arrays)"
        )
        predict_trainer.predict(model, datamodule, return_predictions=False)
        logger.info(f"Predictions saved to {hdf5_path}")

        logger.info(f"Completed fold {fold_idx + 1}/{n_folds}")

    logger.info(f"\n{'=' * 60}")
    logger.info("All folds completed!")
    logger.info(f"Models saved to: {model_dir}")
    logger.info(f"Predictions saved to: {hdf5_path}")
    logger.info(f"{'=' * 60}\n")


if __name__ == "__main__":
    # python -m autotslabel.autosegment.multichannel.step_6b_refiner
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path)
