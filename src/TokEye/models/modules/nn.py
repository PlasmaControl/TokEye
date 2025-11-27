"""
Neural Network Building Blocks for UNet Architecture

This module provides the core building blocks used in the UNet implementation,
including convolutional blocks, downsampling blocks, and upsampling blocks.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Double convolution block with batch normalization and LeakyReLU activation.

    This block applies two 3x3 convolutions, each followed by batch normalization
    and LeakyReLU activation. Optional dropout can be applied after each activation.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        mid_channels: Number of intermediate channels (defaults to out_channels)
        dropout_rate: Dropout probability (0.0 means no dropout)

    Example:
        >>> block = ConvBlock(3, 64, dropout_rate=0.2)
        >>> x = torch.randn(1, 3, 256, 256)
        >>> out = block(x)
        >>> print(out.shape)  # torch.Size([1, 64, 256, 256])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        layers = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
        ]
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(p=dropout_rate))

        layers.extend(
            [
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(inplace=True),
            ]
        )
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(p=dropout_rate))

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the double convolution block.

        Args:
            x: Input tensor of shape (batch, in_channels, height, width)

        Returns:
            Output tensor of shape (batch, out_channels, height, width)
        """
        return self.double_conv(x)


class DownBlock(nn.Module):
    """
    Downsampling block using max pooling followed by convolution.

    This block first applies 2x2 max pooling to reduce spatial dimensions by half,
    then applies a ConvBlock for feature extraction.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        dropout_rate: Dropout probability for the ConvBlock

    Example:
        >>> block = DownBlock(64, 128, dropout_rate=0.2)
        >>> x = torch.randn(1, 64, 256, 256)
        >>> out = block(x)
        >>> print(out.shape)  # torch.Size([1, 128, 128, 128])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_rate=dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the downsampling block.

        Args:
            x: Input tensor of shape (batch, in_channels, height, width)

        Returns:
            Output tensor of shape (batch, out_channels, height/2, width/2)
        """
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """
    Upsampling block using bilinear interpolation with skip connections.

    This block upsamples the input by 2x using bilinear interpolation, concatenates
    with a skip connection from the encoder path, and applies a ConvBlock.

    Args:
        in_channels: Number of channels in the upsampled input
        out_channels: Number of output channels
        dropout_rate: Dropout probability for the ConvBlock

    Example:
        >>> block = UpBlock(128, 64, dropout_rate=0.2)
        >>> x1 = torch.randn(1, 128, 128, 128)  # From decoder
        >>> x2 = torch.randn(1, 64, 256, 256)   # Skip connection
        >>> out = block(x1, x2)
        >>> print(out.shape)  # torch.Size([1, 64, 256, 256])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(
            in_channels + out_channels, out_channels, dropout_rate=dropout_rate
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the upsampling block.

        Args:
            x1: Input tensor from decoder path (batch, in_channels, H, W)
            x2: Skip connection from encoder path (batch, out_channels, H*2, W*2)

        Returns:
            Output tensor of shape (batch, out_channels, H*2, W*2)
        """
        x1 = self.up(x1)

        # Handle size mismatches with padding
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1,
            [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2],
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


if __name__ == "__main__":
    # python -m TokEye.models.modules.nn
    conv_model = ConvBlock(3, 64)
    x = torch.randn(1, 3, 256, 256)
    print(conv_model(x).shape)

    down_model = DownBlock(3, 64)
    x = torch.randn(1, 3, 256, 256)
    print(down_model(x).shape)

    up_model = UpBlock(3, 64)
    x1 = torch.randn(1, 3, 128, 128)
    x2 = torch.randn(1, 64, 256, 256)
    print(up_model(x1, x2).shape)
