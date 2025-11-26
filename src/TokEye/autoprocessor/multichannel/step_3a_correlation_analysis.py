import sys
from pathlib import Path

import joblib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.fp32_precision = 'ieee'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from torchmetrics.image import TotalVariation

import lightning as L
from lightning.pytorch.callbacks import Callback
L.seed_everything(42)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from autotslabel.models.unet import UNet

from .utils.configuration import load_settings, load_input_paths, setup_directory

default_settings = {
    'adjacent_channels': 3,
    'total_channels': 8,
    'clamp_range': (0.01, 1.9),
    'num_layers': 5,
    'first_layer_size': 16,
    'batch_size': 36,
    'num_workers': 8,
    'prefetch_factor': 2,
    'tv_early_stopping': True,
    'tv_patience': 3,
    'max_epochs': 30,
    'precision': 'bf16-mixed',
    'devices': 1,
    'enable_progress_bar': False,
    'ckpt_path': None,
    'log_every_n_steps': 10,
    'input_dir': Path('data/cache/step_2b_filter_spectrogram'),
    'output_dir': Path('data/cache/step_3a_correlation_analysis'),
    'fast_dev_run': False,
    'overwrite': True,
}

class TotalVariationEarlyStopping(Callback):
    """Early stopping callback based on total variation increase"""
    
    def __init__(self, patience=3, min_delta=0.0):
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.best_tv = float('inf')
        self.tv_increase_count = 0
        
    def on_train_epoch_end(self, trainer, pl_module):
        current_tv = trainer.callback_metrics.get('train_tv', None)
        if current_tv is None: return
        current_tv = float(current_tv)
        if current_tv > self.best_tv + self.min_delta:
            self.tv_increase_count += 1
            logger.info(
                f"Current TV: {current_tv:.6f}, Best TV: {self.best_tv:.6f}",
            )
            
            if self.tv_increase_count >= self.patience:
                trainer.should_stop = True
        else:
            self.best_tv = current_tv
            self.tv_increase_count = 0

class ECEDataset(Dataset):
    def __init__(
        self, 
        data_files, 
        transform=None, 
        settings=default_settings,
    ):
        self.data_files = data_files
        self.transform = transform
        self.settings = settings
        print(f"Found {len(self.data_files)} data files")
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        file_path = self.data_files[idx]

        with open(file_path, 'rb') as f:
            data = joblib.load(f)
        data = torch.from_numpy(data).float()
        # dim = C, H, W, Z

        data[:,:4] = .75
        data[:,-3:] = .75

        minval, maxval = self.settings['clamp_range']
        data = data.nan_to_num(0, minval, maxval)
        data = data.clamp(min=minval, max=maxval)

        mean = data.mean(dim=(1,2), keepdim=True)
        std = data.std(dim=(1,2), keepdim=True)
        spectrogram = (data - mean) / (std + 1e-6)
        
        return spectrogram

class ECEDataModule(L.LightningDataModule):
    def __init__(
        self, 
        data_dir, 
        batch_size=4, 
        num_workers=4, 
        prefetch_factor=4,
        settings=default_settings,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.settings = settings
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        
    def setup(self, stage=None):
        self.data_files = list(self.data_dir.glob("*.joblib"))
        self.data_files.sort(key=lambda x: int(x.stem))

        self.dataset = ECEDataset(
            data_files=self.data_files,
            settings=self.settings,
        )
        
    def train_dataloader(self):
        return DataLoader(
            self.dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            prefetch_factor=self.prefetch_factor,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )

class BTN(nn.Module):
    def __init__(self, in_channels=4, num_layers=4, first_layer_size=16):
        super().__init__()
        self.in_channels = in_channels
        self.unet = UNet(
            in_channels=in_channels,
            out_channels=in_channels,
            num_layers=num_layers,
            first_layer_size=first_layer_size,
        )

    def forward(self, x):
        x_real, x_imag = x[..., 0], -x[..., 1]
        x_real = self.unet(x_real)
        x_imag = self.unet(x_imag)
        x = torch.stack([x_real, x_imag], dim=-1)
        return x

class BTNModule(L.LightningModule):
    def __init__(
        self, 
        adjacent_channels=3, 
        total_channels=5, 
        num_layers=4, 
        first_layer_size=16,
        settings=default_settings,
    ):
        super().__init__()
        if adjacent_channels >= total_channels:
            raise ValueError(
                "adjacent_channels must be less than total_channels",
                "adjacent_channels: ", adjacent_channels,
                "total_channels: ", total_channels,
            )
        self.adjacent_channels = adjacent_channels
        self.in_channels = 2 * adjacent_channels
        self.total_channels = total_channels
        self.num_layers = num_layers
        self.first_layer_size = first_layer_size
        self.settings = settings
        self.unet = BTN(
            in_channels=self.in_channels,
            num_layers=num_layers,
            first_layer_size=first_layer_size,
        )
        if not self.settings['fast_dev_run']:
            self.unet.compile()
        
        self.pad_len = self.adjacent_channels
        self.center_channel = self.total_channels // 2 + self.pad_len
        self.padding = nn.ReflectionPad3d(
            (
                0,            0, 
                0,            0, 
                self.pad_len, self.pad_len,
            )
        )

        self.loss_fn = nn.L1Loss()
        self.train_tv = TotalVariation()
        self.predict_tv = TotalVariation()

    def _load_adjacent_channels(self, x, target_channel):
        if target_channel is None: target_channel = self.center_channel
        if target_channel < 0: target_channel = self.total_channels + target_channel
        
        channel_idx = target_channel + self.pad_len
        front_channel_idx = channel_idx - self.pad_len
        back_channel_idx = channel_idx + 1 + self.pad_len

        x_real, x_imag = x[..., 0], x[..., 1]
        x_real = self.padding(x_real)
        x_imag = self.padding(x_imag)
        x = torch.stack([x_real, x_imag], dim=-1)

        front_channels = x[:, front_channel_idx:channel_idx]
        back_channels = x[:, channel_idx + 1:back_channel_idx]
        target_channel = x[:, target_channel:target_channel+1]
        target_channel = torch.rot90(target_channel, k=2, dims=[2, 3])

        x = torch.cat([front_channels, back_channels], dim=1)
        # x[:, 0:0+1] = (target_channel +  x[:, 0:0+1]) / 2
        # x[:, -1:-1+1] = (target_channel + x[:, -1:-1+1]) / 2
        return x

    def _load_target_channels(self, x, target_channel):
        if target_channel is None: target_channel = self.center_channel
        if target_channel < 0: target_channel = self.total_channels + target_channel
        x = x[:, target_channel:target_channel+1]
        x = x.repeat(1, self.in_channels, 1, 1, 1)
        return x

    def _single_channel_loss(self, y_hat, y) -> torch.Tensor:
        loss = self.loss_fn(y_hat, y.flip(-1))
        self.train_tv.update(y_hat[..., 0])
        self.train_tv.update(y_hat[..., 1])
        return loss

    def _multichannel_loss(self, y_hat, y) -> torch.Tensor:
        loss = torch.tensor(0.0, device=y_hat.device)
        for i in range(self.in_channels):
            y_hat_instance = y_hat[:, i:i+1]
            y_instance = y[:, i:i+1]
            loss += self.loss_fn(y_hat_instance, y_instance.flip(-1))
            self.train_tv.update(y_hat_instance[..., 0] / self.in_channels)
            self.train_tv.update(y_hat_instance[..., 1] / self.in_channels)
        return loss / self.in_channels

    def forward(self, x):
        x = self.unet(x)
        return x

    def training_step(self, batch, batch_idx):
        batch = batch

        target_channel = torch.randint(0, self.total_channels, (1,))
        B, C, H, W, Z = batch.shape
        loss = torch.tensor(0.0, device=batch.device)

        x1 = self._load_adjacent_channels(batch, target_channel)
        y1, y_hat1 = batch[:, target_channel:target_channel+1], self(x1)
        y_hat1 = y_hat1.mean(dim=1).unsqueeze(1)
        loss1 = self._single_channel_loss(y_hat1, y1)
        loss += loss1
        self.log('train_loss1', loss1)

        x2 = self._load_target_channels(batch, target_channel)
        y2, y_hat2 = x1, self(x2)
        loss2 = self._multichannel_loss(y_hat2, y2)
        loss += loss2
        self.log('train_loss2', loss2)

        self.log('train_loss', loss)

        return loss

    def on_train_epoch_end(self):
        tv = self.train_tv.compute()
        print(f"Train TV: {tv}")
        self.log('train_tv', tv, logger=True)
        self.train_tv.reset()
    
    def predict_step(self, batch, batch_idx):
        batch = batch

        target_channels = torch.arange(self.total_channels)

        xs = [self._load_adjacent_channels(batch, target_channel)
            for target_channel
            in target_channels]
        ys = [self._load_target_channels(batch, target_channel)
            for target_channel 
            in target_channels]

        y_hats1 = [self(x).mean(dim=1).unsqueeze(1) for x in xs]
        y_hats1 = torch.cat(y_hats1, dim=1)
        y_hats2 = [self(y).mean(dim=1).unsqueeze(1) for y in ys]
        y_hats2 = torch.cat(y_hats2, dim=1)

        out_data = torch.cat([y_hats1, y_hats2], dim=-1).float().cpu().numpy()

        joblib.dump(out_data, self.settings['output_dir'] / f'{batch_idx}.joblib')

        y_hat1_real, y_hat1_imag = y_hats1[..., 0], y_hats1[..., 1]
        y_hat2_real, y_hat2_imag = y_hats2[..., 0], y_hats2[..., 1]
        self.predict_tv.update(y_hat1_real)
        self.predict_tv.update(y_hat1_imag)
        self.predict_tv.update(y_hat2_real)
        self.predict_tv.update(y_hat2_imag)
        return 0
        
    def on_predict_epoch_end(self):
        tv = self.predict_tv.compute()
        logger.info(f"Predict TV: {tv}")
        print(f"Predict TV: {tv}")
        logger.info(f"Predict TV: {tv}")
        self.predict_tv.reset()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=1e-4,
        )
        return optimizer

def main(config_path=None):
    settings = load_settings(config_path, default_settings)
    setup_directory(
        path=settings['output_dir'], 
        overwrite=settings['overwrite'],
        )
    datamodule = ECEDataModule(
        data_dir=settings['input_dir'], 
        batch_size=settings['batch_size'],
        num_workers=settings['num_workers'],
        prefetch_factor=settings['prefetch_factor'],
        settings=settings,
    )
    model = BTNModule(
        adjacent_channels=settings['adjacent_channels'],
        total_channels=settings['total_channels'],
        num_layers=settings['num_layers'],
        first_layer_size=settings['first_layer_size'],
        settings=settings,
    )
    
    # Set up callbacks
    callbacks = []
    if settings['tv_early_stopping']:
        tv_callback = TotalVariationEarlyStopping(
            patience=settings['tv_patience']
            )
        callbacks.append(tv_callback)
    
    trainer = L.Trainer(
        max_epochs=settings['max_epochs'],
        precision=settings['precision'],
        devices=settings['devices'],
        log_every_n_steps=settings['log_every_n_steps'],
        enable_progress_bar=settings['enable_progress_bar'],
        fast_dev_run=settings['fast_dev_run'],
        callbacks=callbacks,
    )

    if settings['ckpt_path'] is not None:
        trainer.fit(model, datamodule, ckpt_path=settings['ckpt_path'])
    else:
        trainer.fit(model, datamodule)

    predictions = trainer.predict(model, datamodule)

if __name__ == "__main__":
    # python -m autotslabel.autosegment.multichannel.step_3a_correlation_analysis
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path)