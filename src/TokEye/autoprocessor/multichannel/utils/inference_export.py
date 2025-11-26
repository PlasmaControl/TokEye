"""
Utilities for exporting inference examples and documentation.
"""

from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def create_inference_example(model_dir: Path):
    """
    Create inference example script demonstrating model usage.
    
    Args:
        model_dir: Directory to save the example script
    """
    logger.info("\n" + "="*60)
    logger.info("Step 5: Creating inference example script")
    logger.info("="*60)
    
    inference_script = '''"""
Inference example for the final segmentation model.
Demonstrates how to load and use both PyTorch Lightning checkpoint and TorchScript model.
"""

import json
from pathlib import Path
import numpy as np
import tifffile as tif
import torch

# Model directory (adjust as needed)
MODEL_DIR = Path(__file__).parent


def load_torchscript_model(model_path=None):
    """
    Load TorchScript model (fastest, most portable).
    No dependencies on Lightning or custom code required.
    
    Args:
        model_path: Path to TorchScript model file
        
    Returns:
        Loaded model
    """
    if model_path is None:
        model_path = MODEL_DIR / 'model.torchscript.pt'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    
    print(f"Loaded TorchScript model from {model_path}")
    return model, device


def load_lightning_checkpoint(checkpoint_path=None):
    """
    Load PyTorch Lightning checkpoint (requires project imports).
    Supports MC Dropout for uncertainty estimation.
    
    Args:
        checkpoint_path: Path to Lightning checkpoint
        
    Returns:
        Loaded model
    """
    if checkpoint_path is None:
        checkpoint_path = MODEL_DIR / 'best_model.ckpt'
    
    # Load model config
    config_path = MODEL_DIR / 'model_config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Import model architecture
    import sys
    sys.path.insert(0, str(MODEL_DIR))
    from unet import UNet
    
    # Load checkpoint
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model = UNet(
        in_channels=config['in_channels'],
        out_channels=config['out_channels'],
        num_layers=config['num_layers'],
        first_layer_size=config['first_layer_size'],
        dropout_rate=config['dropout_rate'],
    )
    
    # Load state dict (strip 'unet.' prefix from Lightning module)
    state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        if key.startswith('unet.'):
            state_dict[key[5:]] = value
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"Loaded Lightning checkpoint from {checkpoint_path}")
    return model, device, config


def predict_single_image(model, image, device='cuda', use_sigmoid=True):
    """
    Run inference on a single image.
    
    Args:
        model: Loaded model
        image: Input image (H, W) or (C, H, W) numpy array
        device: Device to run on
        use_sigmoid: Apply sigmoid to output (set False for TorchScript with sigmoid included)
        
    Returns:
        Prediction as numpy array (H, W)
    """
    # Prepare input
    if image.ndim == 2:
        image = image[np.newaxis, ...]  # Add channel dimension
    
    # Convert to tensor and add batch dimension
    x = torch.from_numpy(image).float().unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        y_pred = model(x)
        if use_sigmoid:
            y_pred = torch.sigmoid(y_pred)
    
    # Convert back to numpy
    pred = y_pred.squeeze().cpu().numpy()
    
    return pred


def predict_with_mc_dropout(model, image, device='cuda', n_samples=15):
    """
    Run inference with MC Dropout for uncertainty estimation.
    Only works with Lightning checkpoint model (not TorchScript).
    
    Args:
        model: Loaded model with dropout layers
        image: Input image (H, W) or (C, H, W) numpy array
        device: Device to run on
        n_samples: Number of MC dropout samples
        
    Returns:
        mean_pred: Mean prediction (H, W)
        std_pred: Standard deviation (epistemic uncertainty) (H, W)
        entropy: Predictive entropy (H, W)
    """
    # Prepare input
    if image.ndim == 2:
        image = image[np.newaxis, ...]
    
    x = torch.from_numpy(image).float().unsqueeze(0).to(device)
    
    # Enable dropout for MC sampling
    model.train()
    
    # Collect predictions
    predictions = []
    with torch.no_grad():
        for _ in range(n_samples):
            y_pred = model(x)
            y_pred = torch.sigmoid(y_pred)
            predictions.append(y_pred)
    
    # Stack and compute statistics
    predictions = torch.stack(predictions, dim=0)  # (n_samples, 1, 1, H, W)
    
    mean_pred = predictions.mean(dim=0).squeeze().cpu().numpy()
    std_pred = predictions.std(dim=0).squeeze().cpu().numpy()
    
    # Compute entropy
    eps = 1e-7
    mean_clipped = np.clip(mean_pred, eps, 1-eps)
    entropy = -(mean_clipped * np.log(mean_clipped) + 
                (1 - mean_clipped) * np.log(1 - mean_clipped))
    
    # Disable dropout
    model.eval()
    
    return mean_pred, std_pred, entropy


def example_torchscript_inference():
    """Example: Simple inference using TorchScript model."""
    print("\\n" + "="*60)
    print("Example 1: TorchScript Inference")
    print("="*60)
    
    # Load model
    model, device = load_torchscript_model()
    
    # Create dummy image (replace with actual image loading)
    dummy_image = np.random.randn(512, 512).astype(np.float32)
    
    # Run inference
    prediction = predict_single_image(model, dummy_image, device, use_sigmoid=True)
    
    print(f"Input shape: {dummy_image.shape}")
    print(f"Prediction shape: {prediction.shape}")
    print(f"Prediction range: [{prediction.min():.3f}, {prediction.max():.3f}]")
    
    # Threshold prediction
    binary_mask = (prediction > 0.5).astype(np.uint8)
    print(f"Positive pixels: {binary_mask.sum()} / {binary_mask.size}")
    
    # Save prediction (optional)
    # tif.imwrite('prediction.tif', prediction)
    

def example_lightning_inference_with_uncertainty():
    """Example: Inference with uncertainty using Lightning checkpoint."""
    print("\\n" + "="*60)
    print("Example 2: Lightning Checkpoint with MC Dropout Uncertainty")
    print("="*60)
    
    # Load model
    model, device, config = load_lightning_checkpoint()
    
    # Create dummy image
    dummy_image = np.random.randn(512, 512).astype(np.float32)
    
    # Run MC Dropout inference
    n_samples = config.get('mc_dropout_samples', 15)
    mean_pred, std_pred, entropy = predict_with_mc_dropout(
        model, dummy_image, device, n_samples
    )
    
    print(f"Input shape: {dummy_image.shape}")
    print(f"Mean prediction shape: {mean_pred.shape}")
    print(f"Mean prediction range: [{mean_pred.min():.3f}, {mean_pred.max():.3f}]")
    print(f"Uncertainty (std) range: [{std_pred.min():.3f}, {std_pred.max():.3f}]")
    print(f"Entropy range: [{entropy.min():.3f}, {entropy.max():.3f}]")
    
    # Save predictions (optional)
    # tif.imwrite('mean_prediction.tif', mean_pred)
    # tif.imwrite('uncertainty_std.tif', std_pred)
    # tif.imwrite('uncertainty_entropy.tif', entropy)


def example_batch_processing():
    """Example: Process multiple images efficiently."""
    print("\\n" + "="*60)
    print("Example 3: Batch Processing")
    print("="*60)
    
    # Load TorchScript model for fastest inference
    model, device = load_torchscript_model()
    
    # Simulate batch of images
    batch_size = 8
    images = np.random.randn(batch_size, 1, 512, 512).astype(np.float32)
    
    # Process batch
    x = torch.from_numpy(images).float().to(device)
    with torch.no_grad():
        predictions = model(x)
        predictions = torch.sigmoid(predictions)
    
    predictions = predictions.cpu().numpy()
    
    print(f"Processed {batch_size} images in one batch")
    print(f"Input shape: {images.shape}")
    print(f"Output shape: {predictions.shape}")


if __name__ == "__main__":
    print("Final Segmentation Model - Inference Examples")
    print("=" * 60)
    
    # Run examples
    try:
        example_torchscript_inference()
    except Exception as e:
        print(f"TorchScript example failed: {e}")
    
    try:
        example_lightning_inference_with_uncertainty()
    except Exception as e:
        print(f"Lightning example failed: {e}")
    
    try:
        example_batch_processing()
    except Exception as e:
        print(f"Batch processing example failed: {e}")
    
    print("\\n" + "="*60)
    print("Examples complete!")
    print("="*60)
'''
    
    # Write script to file
    script_path = model_dir / 'inference_example.py'
    with open(script_path, 'w') as f:
        f.write(inference_script)
    
    logger.info(f"✓ Created inference example: {script_path}")
    logger.info(f"  Run with: python {script_path}")

