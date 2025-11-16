"""
Model Inference Utilities

This module provides utilities for loading PyTorch models and running
batch inference on tiled spectrograms.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Union, Literal
from pathlib import Path
import warnings


def load_model(
    model_path: str,
    device: str = 'auto',
    map_location: Optional[str] = None,
) -> Union[nn.Module, torch.export.ExportedProgram]:
    """
    Load a PyTorch model for inference (supports TorchScript .pt or torch.export .pt2).

    Args:
        model_path: Path to the model file (.pt for TorchScript, .pt2 for torch.export)
        device: Device to load model on ('auto', 'cuda', 'cpu', or specific device like 'cuda:0')
                If 'auto', will use CUDA if available, otherwise CPU
        map_location: Optional torch device string for loading checkpoint
                     If None, will be set based on device parameter

    Returns:
        Loaded PyTorch model or ExportedProgram in evaluation mode

    Raises:
        FileNotFoundError: If model_path doesn't exist
        RuntimeError: If model loading fails
        ValueError: If device string is invalid

    Example:
        >>> model = load_model('model.pt2', device='auto')  # torch.export model
        >>> model = load_model('model.pt', device='auto')   # TorchScript model
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Check model type based on extension
    model_suffix = model_path.suffix
    print(f"Detected model file extension: {model_suffix}")
    if model_suffix not in ['.pt', '.pt2']:
        warnings.warn(
            f"Model file extension is '{model_suffix}', expected '.pt' or '.pt2'. "
            f"Attempting to load anyway.",
            RuntimeWarning
        )

    # Determine device
    if device == 'auto':
        target_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        target_device = device

    # Validate device
    if target_device.startswith('cuda'):
        if not torch.cuda.is_available():
            warnings.warn(
                f"CUDA device requested but CUDA is not available. "
                f"Falling back to CPU.",
                RuntimeWarning
            )
            target_device = 'cpu'
        elif ':' in target_device:
            # Validate specific CUDA device
            device_id = int(target_device.split(':')[1])
            if device_id >= torch.cuda.device_count():
                raise ValueError(
                    f"CUDA device {device_id} not available. "
                    f"Available devices: 0-{torch.cuda.device_count()-1}"
                )

    # Set map_location if not provided
    if map_location is None:
        map_location = target_device

    # Load model based on type
    try:
        if model_suffix == '.pt2':
            # Load torch.export model
            print(f"Loading torch.export model from {model_path}")
            module = torch.export.load(str(model_path))
            model = module.module()

            # For torch.export, we need to move to device differently
            # The ExportedProgram needs to be used with torch.export.execute
            print(f"torch.export model loaded successfully")

        else:  # .pt or other
            # Load TorchScript model
            print(f"Loading TorchScript model from {model_path}")
            model = torch.jit.load(str(model_path), map_location=map_location)

            # Move to target device
            model = model.to(target_device)

            # Set to evaluation mode
            model.eval()

    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

    # Print device info
    device_name = torch.cuda.get_device_name(0) if target_device.startswith('cuda') else 'CPU'
    print(f"Model loaded on {target_device} ({device_name})")

    return model


def batch_inference(
    model: nn.Module,
    tiles: List[np.ndarray],
    batch_size: int = 32,
    device: Optional[str] = None,
    show_progress: bool = False,
    dtype: torch.dtype = torch.float32,
) -> List[np.ndarray]:
    """
    Run batch inference on a list of tiles.

    Args:
        model: PyTorch model in evaluation mode
        tiles: List of input tiles as numpy arrays
        batch_size: Number of tiles to process per batch
        device: Device to run inference on (if None, uses model's device)
        show_progress: If True, print progress information
        dtype: Data type for torch tensors (default: float32)

    Returns:
        List of prediction arrays as numpy arrays, same length as input tiles

    Raises:
        ValueError: If tiles list is empty
        ValueError: If tile shapes are inconsistent
        RuntimeError: If inference fails

    Example:
        >>> model = load_model('model.pt')
        >>> tiles = [np.random.randn(1, 256, 256) for _ in range(100)]
        >>> predictions = batch_inference(model, tiles, batch_size=32)
        >>> print(len(predictions))  # 100
    """
    if not tiles:
        raise ValueError("Tiles list cannot be empty")

    # Validate tile shapes are consistent
    first_shape = tiles[0].shape
    for i, tile in enumerate(tiles):
        if tile.shape != first_shape:
            raise ValueError(
                f"Tile {i} has shape {tile.shape}, expected {first_shape}. "
                f"All tiles must have the same shape."
            )

    # Determine device
    if device is None:
        # Get device from model parameters
        try:
            device = next(model.parameters()).device
        except StopIteration:
            # Model has no parameters, default to CPU
            device = torch.device('cpu')
    else:
        device = torch.device(device)

    num_tiles = len(tiles)
    predictions = []

    # Disable gradient computation for inference
    with torch.no_grad():
        # Process in batches
        for batch_start in range(0, num_tiles, batch_size):
            batch_end = min(batch_start + batch_size, num_tiles)
            batch_tiles = tiles[batch_start:batch_end]

            if show_progress:
                print(f"Processing batch {batch_start//batch_size + 1}/{(num_tiles + batch_size - 1)//batch_size}")

            # Stack tiles into batch
            try:
                batch_array = np.stack(batch_tiles, axis=0)
            except Exception as e:
                raise RuntimeError(f"Failed to stack tiles into batch: {e}")

            # Convert to torch tensor
            try:
                batch_tensor = torch.from_numpy(batch_array).to(device=device, dtype=dtype)
            except Exception as e:
                raise RuntimeError(f"Failed to convert batch to torch tensor: {e}")

            # Run inference
            try:
                batch_predictions = model(batch_tensor)
            except Exception as e:
                raise RuntimeError(f"Model inference failed: {e}")

            # Convert predictions back to numpy
            try:
                # Handle both tensor and tuple/list outputs
                if isinstance(batch_predictions, (tuple, list)):
                    batch_predictions = batch_predictions[0]

                batch_predictions_np = batch_predictions.cpu().numpy()
            except Exception as e:
                raise RuntimeError(f"Failed to convert predictions to numpy: {e}")

            # Split batch back into individual predictions
            for i in range(len(batch_tiles)):
                predictions.append(batch_predictions_np[i])

    if len(predictions) != num_tiles:
        raise RuntimeError(
            f"Number of predictions ({len(predictions)}) doesn't match "
            f"number of input tiles ({num_tiles})"
        )

    return predictions


def get_model_info(model: nn.Module) -> dict:
    """
    Get information about a loaded model.

    Args:
        model: PyTorch model

    Returns:
        Dictionary containing model information:
        - 'device': Device model is on
        - 'dtype': Data type of model parameters
        - 'num_parameters': Total number of parameters
        - 'trainable_parameters': Number of trainable parameters

    Example:
        >>> model = load_model('model.pt')
        >>> info = get_model_info(model)
        >>> print(f"Model has {info['num_parameters']:,} parameters")
    """
    # Get device
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = None

    # Get dtype
    try:
        dtype = next(model.parameters()).dtype
    except StopIteration:
        dtype = None

    # Count parameters
    num_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'device': str(device) if device else 'unknown',
        'dtype': str(dtype) if dtype else 'unknown',
        'num_parameters': num_parameters,
        'trainable_parameters': trainable_parameters,
    }


def warmup_model(
    model: nn.Module,
    input_shape: tuple,
    num_iterations: int = 10,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
) -> float:
    """
    Warm up model with dummy inputs to initialize CUDA kernels.

    This can improve inference performance by pre-compiling CUDA kernels
    before processing real data.

    Args:
        model: PyTorch model
        input_shape: Shape of input tensor (including batch dimension)
        num_iterations: Number of warmup iterations
        device: Device to run on (if None, uses model's device)
        dtype: Data type for dummy input

    Returns:
        Average inference time per iteration (in seconds)

    Example:
        >>> model = load_model('model.pt', device='cuda')
        >>> avg_time = warmup_model(model, input_shape=(1, 1, 256, 256))
        >>> print(f"Warmup complete. Avg time: {avg_time*1000:.2f}ms")
    """
    import time

    # Determine device
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device('cpu')
    else:
        device = torch.device(device)

    # Create dummy input
    dummy_input = torch.randn(*input_shape, device=device, dtype=dtype)

    # Warmup iterations
    times = []
    with torch.no_grad():
        for i in range(num_iterations):
            start_time = time.time()
            _ = model(dummy_input)

            # Synchronize if using CUDA
            if device.type == 'cuda':
                torch.cuda.synchronize()

            end_time = time.time()
            times.append(end_time - start_time)

    avg_time = np.mean(times)
    return avg_time
