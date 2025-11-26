import sys

from pathlib import Path
import shutil

import tifffile as tif
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
torch.backends.cuda.matmul.fp32_precision = 'ieee'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
L.seed_everything(42)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .utils.configuration import load_settings
from .utils.augmentations import get_augmentation
from .utils.losses import get_loss_function, dice_coefficient, iou_score

from TokEye.models.unet import UNet

default_settings = {
    # Data and training
    'batch_size': 56,
    'num_workers': 6,
    'prefetch_factor': 2,
    'num_layers': 5,
    'first_layer_size': 32,
    'max_epochs': 100,
    'precision': 'bf16-mixed',
    'devices': 1,
    'enable_progress_bar': False,
    'ckpt_path': None,
    'log_every_n_steps': 10,
    'input_dir': Path('data/cache/step_6c_refined_masks'),
    'model_dir': Path('models/final'),
    'fast_dev_run': False,
    'early_stopping_patience': 3,
    
    # Augmentation settings
    'augmentation': True,
    'aug_rotation_degrees': 0,
    'aug_prob_flip': 0.5,
    'aug_elastic': False,
    'aug_elastic_alpha': 50.0,
    'aug_elastic_sigma': 5.0,
    'aug_scale_range': [0.8, 1.2],
    'aug_intensity': True,
    'aug_brightness_range': [0.8, 1.2],
    'aug_contrast_range': [0.8, 1.2],
    'aug_noise_std': 0.05,
    'aug_blur_prob': 0.01,
    'aug_blur_sigma_range': [0.9, 1.1],
    'aug_gamma_range': [0.9, 1.1],
    'aug_apply_prob': 0.03,
    
    # SpecAugment settings (for spectrogram-specific augmentation)
    'specaugment': False,
    'specaug_time_warp_W': 20,  # time warp parameter W (max displacement)
    'specaug_freq_mask_F': 5,  # max frequency mask width
    'specaug_time_mask_T': 5,  # max time mask width
    'specaug_freq_mask_num': 2,  # number of frequency masks (0=disabled by default)
    'specaug_time_mask_num': 2,  # number of time masks (0=disabled by default)
    
    # Loss function settings
    'loss_type': 'focal',  # Options: 'bce', 'label_smooth_bce', 'symmetric_bce', 'dice', 'dice_bce', 'symmetric_bce_dice', 'focal', 'focal_dice', 'iou'
    'label_smoothing': 0.1,
    'symmetric_alpha': 0.1,
    'symmetric_beta': 1.0,
    'symmetric_weight': 0.5,
    'dice_weight': 0.5,
    'bce_weight': 0.5,
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,
    'focal_weight': 0.5,
    
    # Optimizer settings
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    
    # Learning rate scheduler
    'lr_scheduler': 'reduce_on_plateau',  # Options: None, 'reduce_on_plateau', 'cosine'
    'lr_factor': 0.5,
    'lr_patience': 5,
    'lr_min': 1e-6,
    
    # MC Dropout
    'dropout_rate': 0.2,
}


class TiffDataset(Dataset):
    def __init__(self, data_files, transform=None, settings=None):
        self.data_files = data_files
        self.transform = transform
        self.settings = settings if settings is not None else {}

    def _get_label_path(self, input_path):
        return input_path.with_name(input_path.name.replace('_img.tif', '_mask.tif'))
    
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
            p for p in self.data_dir.glob('*.tif') 
            if '_mask' not in p.name
        ]
        self.data_files.sort(key=lambda x: int(x.stem.replace('_img', '')))
        
        # Create base datasets with appropriate transforms
        train_augmentation = get_augmentation(self.settings)
        full_train_dataset = TiffDataset(
            data_files=self.data_files, 
            transform=train_augmentation,
            settings=self.settings
        )
        full_val_dataset = TiffDataset(
            data_files=self.data_files, 
            transform=None,
            settings=self.settings
        )
        
        # Create subsets for train/val splits
        self.train_dataset = Subset(full_train_dataset, self.train_indices)
        self.val_dataset = Subset(full_val_dataset, self.val_indices)
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            prefetch_factor=self.prefetch_factor,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=False,
            prefetch_factor=self.prefetch_factor,
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
            dropout_rate=settings.get('dropout_rate', 0.0),
        )
        if not self.settings['fast_dev_run']:
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
        y_hat = self(x)[0]
        
        # Channel names for logging
        channel_names = ['normal', 'baseline']
        
        # Compute loss and metrics per channel
        losses = []
        dice_scores = []
        iou_scores = []
        
        for ch_idx, ch_name in enumerate(channel_names):
            # Extract channel predictions and targets
            y_hat_ch = y_hat[:, ch_idx:ch_idx+1, ...]
            y_ch = y[:, ch_idx:ch_idx+1, ...]
            
            # Compute metrics
            loss_ch = self.loss_fn(y_hat_ch, y_ch)
            dice_ch = dice_coefficient(y_hat_ch, y_ch)
            iou_ch = iou_score(y_hat_ch, y_ch)
            
            losses.append(loss_ch)
            dice_scores.append(dice_ch)
            iou_scores.append(iou_ch)
            
            # Log per-channel metrics
            self.log(f'{stage}_loss_{ch_name}', loss_ch)
            self.log(f'{stage}_dice_{ch_name}', dice_ch)
            self.log(f'{stage}_iou_{ch_name}', iou_ch)
        
        # Compute average metrics
        total_loss = sum(losses) / len(losses)
        avg_dice = sum(dice_scores) / len(dice_scores)
        avg_iou = sum(iou_scores) / len(iou_scores)
        
        # Log average metrics
        self.log(f'{stage}_loss', total_loss, prog_bar=True)
        self.log(f'{stage}_dice', avg_dice, prog_bar=True)
        self.log(f'{stage}_iou', avg_iou)
        
        return total_loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, 'val')
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.settings.get('learning_rate', 1e-4),
            weight_decay=self.settings.get('weight_decay', 1e-5),
        )
        
        lr_scheduler_type = self.settings.get('lr_scheduler', None)
        
        if lr_scheduler_type == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.settings.get('lr_factor', 0.5),
                patience=self.settings.get('lr_patience', 5),
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'interval': 'epoch',
                    'frequency': 1,
                }
            }
        elif lr_scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.settings.get('max_epochs', 40),
                eta_min=self.settings.get('lr_min', 1e-6),
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1,
                }
            }
        else:
            return optimizer


def main(config_path=None):
    if config_path is None: 
        settings = default_settings
    else: 
        settings = load_settings(config_path)
    
    input_dir = settings['input_dir']
    model_dir = settings['model_dir']
    
    # Create model directory
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all data files
    data_files = [p for p in input_dir.glob('*.tif') if '_mask' not in p.name]
    data_files.sort(key=lambda x: int(x.stem.replace('_img', '')))
    n_samples = len(data_files)
    
    logger.info(f"Found {n_samples} data files")
    logger.info(f"Training final model on all data")
    
    # Use all data for training (no validation split)
    all_indices = list(range(n_samples))
    
    # Create data module with all data for training
    datamodule = DataModule(
        data_dir=input_dir,
        batch_size=settings['batch_size'],
        num_workers=settings['num_workers'],
        prefetch_factor=settings['prefetch_factor'],
        train_indices=all_indices,
        val_indices=all_indices,  # Use same indices for validation
        settings=settings,
    )
    
    # Create model
    model = Module(
        num_layers=settings['num_layers'],
        first_layer_size=settings['first_layer_size'],
        settings=settings,
    )
    
    # Set up callbacks for training
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir,
        filename='best_model',
        monitor='val_loss',
        mode='min',
        save_top_k=5,
    )
    
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=settings['early_stopping_patience'],
        mode='min',
    )
    
    train_callbacks = [checkpoint_callback, early_stopping_callback]
    
    # Create trainer for training
    trainer = L.Trainer(
        max_epochs=settings['max_epochs'],
        precision=settings['precision'],
        devices=settings['devices'],
        log_every_n_steps=settings['log_every_n_steps'],
        enable_progress_bar=settings['enable_progress_bar'],
        fast_dev_run=settings['fast_dev_run'],
        callbacks=train_callbacks,
    )
    
    # Train
    logger.info(f"\n{'='*60}")
    logger.info(f"Training final model on {n_samples} samples")
    logger.info(f"{'='*60}\n")
    trainer.fit(model, datamodule)
    
    # Load best model for export
    best_model_path = model_dir / 'best_model.ckpt'
    if best_model_path.exists():
        logger.info(f"Loading best model from {best_model_path}")
        model = Module.load_from_checkpoint(
            best_model_path,
            num_layers=settings['num_layers'],
            first_layer_size=settings['first_layer_size'],
            settings=settings,
        )
    
    # Export to TorchScript
    logger.info("Exporting model to TorchScript...")
    model = model.to(device)
    model.eval()
    model.unet.eval()
    
    # Get sample input size from first data file
    sample_img = tif.imread(data_files[0])
    sample_shape = sample_img.shape  # Should be (1, H, W)
    if len(sample_shape) == 2:
        sample_shape = (1,) + sample_shape
    
    # Create example input tensor
    example_input = torch.randn(1, *sample_shape).to(device)
    
    # Trace the model
    with torch.no_grad():
        script = torch.jit.trace(model.unet, example_input)
    
    # Save TorchScript model
    torchscript_path = model_dir / 'final.torchscript.pt'
    torch.jit.save(script, str(torchscript_path))
    
    logger.info(f"\n{'='*60}")
    logger.info("Training completed!")
    logger.info(f"Model checkpoint saved to: {best_model_path}")
    logger.info(f"TorchScript model saved to: {torchscript_path}")
    logger.info(f"{'='*60}\n")

if __name__ == "__main__":
    # python -m autotslabel.autosegment.multichannel.step_6d_final
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path)