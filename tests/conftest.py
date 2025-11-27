"""
Pytest configuration and shared fixtures for TokEye tests.

This module provides common fixtures used across all test modules, including
sample signals, spectrograms, and mock model instances.
"""

from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture
def sample_signal_1d() -> np.ndarray:
    """
    Generate a simple 1D test signal.

    Returns:
        1D numpy array of shape (1000,) containing a sine wave with noise
    """
    t = np.linspace(0, 1, 1000)
    signal = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(1000)
    return signal.astype(np.float32)


@pytest.fixture
def sample_signal_2d() -> np.ndarray:
    """
    Generate a 2D test signal (multi-channel).

    Returns:
        2D numpy array of shape (3, 1000) containing 3 channels of sine waves
    """
    t = np.linspace(0, 1, 1000)
    signals = np.zeros((3, 1000), dtype=np.float32)
    for i in range(3):
        freq = 10 + i * 5
        signals[i] = np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(1000)
    return signals


@pytest.fixture
def sample_spectrogram_small() -> np.ndarray:
    """
    Generate a small test spectrogram.

    Returns:
        2D numpy array of shape (128, 256) simulating a spectrogram
    """
    np.random.seed(42)
    spec = np.random.rand(128, 256).astype(np.float32)
    return spec


@pytest.fixture
def sample_spectrogram_256() -> np.ndarray:
    """
    Generate a 256x256 test spectrogram (standard tile size).

    Returns:
        2D numpy array of shape (256, 256) simulating a square spectrogram
    """
    np.random.seed(42)
    spec = np.random.rand(256, 256).astype(np.float32)
    return spec


@pytest.fixture
def sample_spectrogram_large() -> np.ndarray:
    """
    Generate a large test spectrogram requiring tiling.

    Returns:
        2D numpy array of shape (256, 1024) simulating a wide spectrogram
    """
    np.random.seed(42)
    spec = np.random.rand(256, 1024).astype(np.float32)
    return spec


@pytest.fixture
def sample_mask() -> np.ndarray:
    """
    Generate a binary mask for testing post-processing.

    Returns:
        2D numpy array of shape (256, 256) with random binary values
    """
    np.random.seed(42)
    mask = (np.random.rand(256, 256) > 0.7).astype(np.uint8)
    return mask


@pytest.fixture
def mock_model() -> torch.nn.Module:
    """
    Create a mock UNet model for testing inference.

    Returns:
        Simple PyTorch model that returns random predictions

    """

    class MockUNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 1, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Return predictions with same spatial dimensions
            batch_size = x.shape[0]
            return torch.sigmoid(self.conv(x))

    model = MockUNet()
    model.eval()
    return model


@pytest.fixture
def temp_model_path(tmp_path: Path) -> Path:
    """
    Create a temporary directory for model files.

    Args:
        tmp_path: Pytest's built-in temporary directory fixture

    Returns:
        Path to a temporary model file location
    """
    model_path = tmp_path / "test_model.pt"
    return model_path


@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Path:
    """
    Create a temporary cache directory for testing.

    Args:
        tmp_path: Pytest's built-in temporary directory fixture

    Returns:
        Path to a temporary cache directory
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """
    Reset random seeds before each test for reproducibility.

    This fixture runs automatically before each test to ensure
    consistent random number generation.
    """
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
