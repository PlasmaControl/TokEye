"""
UNet Architecture for Image Segmentation

This module implements a UNet architecture with configurable depth and channels,
designed for spectrogram segmentation tasks in TokEye.
"""

import torch
import torch.nn as nn

from .nn import (
    TokEyeConvBlock as ConvBlock,
)
from .nn import (
    TokEyeDownBlock as DownBlock,
)
from .nn import (
    TokEyeUpBlock as UpBlock,
)


class UNet(nn.Module):

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
        layer_sizes: list[int] = [
            first_layer_size * 2**i for i in range(self.num_layers)
        ]

        # Initial convolution block
        self.in_conv = ConvBlock(
            in_channels,
            layer_sizes[0],
            dropout_rate=dropout_rate,
        )

        # Encoder path (downsampling)
        encoder: list[DownBlock] = []
        for i in range(self.num_layers - 1):
            in_ch = layer_sizes[i]
            out_ch = layer_sizes[i + 1]
            encoder.append(DownBlock(in_ch, out_ch, dropout_rate=dropout_rate))
        self.encoder = nn.ModuleList(encoder)

        # Decoder path (upsampling)
        decoder: list[UpBlock] = []
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

    def forward(self, in_BCHW: torch.Tensor) -> tuple[torch.Tensor]:
        skip_BCHW: list[torch.Tensor] = []

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
        return self.out_conv(decode_BCHW)


def build_model(
    num_layers: int = 5,
    first_layer_size: int = 32,
    dropout_rate: float = 0.2,
):
    return UNet(
            in_channels=1,
            out_channels=2,  # 2 channels: normal (ch0) and baseline (ch1)
            num_layers=num_layers,
            first_layer_size=first_layer_size,
            dropout_rate=dropout_rate,
        )

def load_model(
    model_path: str,
    device: str = "auto",
):
    model = build_model()
    model.load_state_dict(torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    ))
    return model


if __name__ == "__main__":
    # python -m TokEye.models.unet
    import torch
    from torchinfo import summary

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(first_layer_size=16)
    input_size = (2, 1, 513, 516)
    dtype = torch.float32

    summary(model, input_size=input_size, dtypes=[dtype], device=device)

    with torch.no_grad():
        output = model(torch.randn(input_size).to(device))
        print(f"Output shape: {output.shape}")
