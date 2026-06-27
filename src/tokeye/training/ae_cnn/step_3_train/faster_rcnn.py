from pathlib import Path

import torch

import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)

from aemodes.utils.dataset import COCODataModule

from aemodes.models.instance.faster_rcnn import build_model

torch.set_float32_matmul_precision('high')

default_settings = {
    'data_path': 'data/.cache/step_2_coco',
    'model_save_path': 'model/faster_rcnn.pt',
    'num_epochs': 30,
    'batch_size': 48,
    'learning_rate': 0.001,
    'num_classes': 2,  # background + 1 class
    'num_workers': 2,
    'precision': 'bf16-mixed',
    'fast_dev_run': False,
}
class FasterRCNNModule(L.LightningModule):
    """Lightning module for Faster R-CNN training."""
    
    def __init__(self, num_classes=2, learning_rate=0.005):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        
        self.model = build_model(num_classes=num_classes)
    
    def forward(self, images, targets=None):
        return self.model(images, targets)
    
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        images, targets = batch
        images = list(images)
        targets = [{k: v for k, v in t.items()} for t in targets]
        
        # Model returns loss dict during training
        loss_dict = self.model(images, targets)
        total_loss = torch.stack(list(loss_dict.values())).sum()
        
        self.log(
            'train_loss', total_loss,
            on_step=True, on_epoch=True, prog_bar=False,
            sync_dist=True, batch_size=len(images),
        )
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        images = list(images)
        
        # Get predictions
        predictions = self.model(images)
        
        # Count detections for simple metrics
        total_gt = sum(len(t['boxes']) for t in targets)
        total_pred = sum(len(p['boxes']) for p in predictions)
        
        return {
            'val_gt_boxes': total_gt,
            'val_pred_boxes': total_pred,
        }

    def test_step(self, batch, batch_idx):
        images, targets = batch
        images = list(images)
        
        # Get predictions
        predictions = self.model(images)
        
        return predictions

    def predict_step(self, batch, batch_idx):
        images, targets = batch
        images = list(images)
        
        # Get predictions
        predictions = self.model(images)
        return predictions
    
    def configure_optimizers(self):  # type: ignore[override]
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
            lr=self.learning_rate,
            weight_decay=0.0005,
        )
        
        # Use cosine annealing - much gentler than aggressive StepLR
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs or 30,
            eta_min=1e-6,
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }


def train_faster_rcnn(settings=None):
    """Main training function using Lightning."""
    
    if settings is None:
        settings = default_settings
    
    # Create data module
    data_module = COCODataModule(
        data_path=settings['data_path'],
        batch_size=settings['batch_size'],
        num_workers=settings['num_workers'],
    )
    
    # Create model
    model = FasterRCNNModule(
        num_classes=settings['num_classes'],
        learning_rate=settings['learning_rate'],
    )
    
    # Setup callbacks
    save_path = Path(settings['model_save_path'])
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_path.parent,
        filename='faster_rcnn-{epoch:02d}-{train_loss:.4f}',
        save_top_k=1,
        monitor='train_loss',
        mode='min',
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Create trainer
    trainer = L.Trainer(
        max_epochs=settings['num_epochs'],
        accelerator='auto',
        devices='auto',
        callbacks=[checkpoint_callback, lr_monitor],
        enable_progress_bar=True,
        log_every_n_steps=10,
        precision=settings['precision'],
        fast_dev_run=settings['fast_dev_run'],
    )
    
    # Print dataset info
    data_module.setup('fit')
    print(f"Train samples: {len(data_module.train_dataset)}")
    print(f"Valid samples: {len(data_module.valid_dataset)}")
    
    # Train
    trainer.fit(model, data_module)
    
    # Save final model weights in the original format for compatibility
    torch.save(model.model.state_dict(), save_path)
    print(f"Saved final model to {save_path}")
    
    print("Training completed!")
    return model


if __name__ == '__main__':
    # python -m aemodes.models.instance.faster_rcnn
    train_faster_rcnn()
