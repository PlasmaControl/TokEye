"""
UNet Architecture for Image Segmentation

This module implements a UNet architecture with configurable depth and channels,
designed for spectrogram segmentation tasks in TokEye.
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from TokEye.models.modules.nn import ConvBlock, DownBlock, UpBlock


class UNet(nn.Module):
    """
    UNet architecture with encoder-decoder structure and skip connections.

    This implementation uses a symmetric encoder-decoder architecture with skip
    connections between corresponding levels. The network progressively downsamples
    features in the encoder and upsamples in the decoder, combining features via
    concatenation at each level.

    Args:
        in_channels: Number of input channels (default: 3 for RGB-like input)
        out_channels: Number of output channels (default: 1 for binary segmentation)
        num_layers: Number of downsampling/upsampling layers (default: 4)
        first_layer_size: Number of filters in the first layer (default: 16).
                         Subsequent layers double this value at each level.
        dropout_rate: Dropout probability applied after activations (default: 0.0)

    Example:
        >>> model = UNet(in_channels=1, out_channels=2, num_layers=4)
        >>> x = torch.randn(2, 1, 256, 256)
        >>> logits, = model(x)
        >>> print(logits.shape)  # torch.Size([2, 2, 256, 256])

    Notes:
        - Input dimensions should be divisible by 2^(num_layers-1) to avoid issues
        - Returns a tuple containing logits for compatibility with loss functions
        - Uses LeakyReLU activation and batch normalization throughout
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        num_layers: int = 4,
        first_layer_size: int = 16,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.first_layer_size = first_layer_size
        self.dropout_rate = dropout_rate

        # Calculate filter sizes for each layer (doubles at each level)
        layer_sizes: List[int] = [
            first_layer_size * 2**i for i in range(self.num_layers)
        ]

        # Initial convolution block
        self.in_conv = ConvBlock(
            in_channels,
            layer_sizes[0],
            dropout_rate=dropout_rate,
        )

        # Encoder path (downsampling)
        encoder: List[DownBlock] = []
        for i in range(self.num_layers - 1):
            in_ch = layer_sizes[i]
            out_ch = layer_sizes[i + 1]
            encoder.append(DownBlock(in_ch, out_ch, dropout_rate=dropout_rate))
        self.encoder = nn.ModuleList(encoder)

        # Decoder path (upsampling)
        decoder: List[UpBlock] = []
        for i in range(self.num_layers - 1):
            in_ch = layer_sizes[-i - 1]
            out_ch = layer_sizes[-i - 2]
            decoder.append(UpBlock(in_ch, out_ch, dropout_rate=dropout_rate))
        self.decoder = nn.ModuleList(decoder)

        # Final 1x1 convolution to produce output channels
        self.out_conv = nn.Conv2d(
            layer_sizes[0],
            out_channels,
            kernel_size=1,
        )

    def forward(self, in_BCHW: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Forward pass through the UNet.

        Args:
            in_BCHW: Input tensor of shape (batch, in_channels, height, width)

        Returns:
            Tuple containing single element:
            - logits: Output tensor of shape (batch, out_channels, height, width)
                     Raw logits before activation (apply sigmoid/softmax externally)

        Example:
            >>> model = UNet(in_channels=1, out_channels=1)
            >>> x = torch.randn(4, 1, 256, 256)
            >>> logits, = model(x)
            >>> probabilities = torch.sigmoid(logits)
        """
        skip_BCHW: List[torch.Tensor] = []

        # Initial convolution
        encode_BCHW = self.in_conv(in_BCHW)
        skip_BCHW.append(encode_BCHW)

        # Encoder path
        for layer in self.encoder:
            encode_BCHW = layer(encode_BCHW)
            skip_BCHW.append(encode_BCHW)

        # Start decoder with bottleneck features
        decode_BCHW = encode_BCHW

        # Decoder path with skip connections
        for i, layer in enumerate(self.decoder):
            skip_idx = len(skip_BCHW) - i - 2
            decode_BCHW = layer(
                decode_BCHW,
                skip_BCHW[skip_idx],
            )

        # Final 1x1 convolution
        logits = self.out_conv(decode_BCHW)

        return (logits,)


if __name__ == "__main__":
    # python -m TokEye.models.unet
    import torch
    from torchinfo import summary  # type: ignore[import-not-found]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(
        in_channels=1,
        out_channels=2,
        num_layers=5,
        first_layer_size=16,
        dropout_rate=0.2,
    )
    input_size = (2, 1, 513, 516)
    dtype = torch.float32

    summary(model, input_size=input_size, dtypes=[dtype], device=device)

    with torch.no_grad():
        (output,) = model(torch.randn(input_size).to(device))
        print(f"Output shape: {output.shape}")
