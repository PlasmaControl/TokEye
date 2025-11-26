import torch.nn as nn

from autotslabel.models.modules.nn import (
    ConvBlock,
    DownBlock,
    UpBlock,
)


class UNet(nn.Module):
    """
    Shared Encoder and Shared Decoder with optional feature extraction for contrastive learning.
    """
    
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        num_layers=4,
        first_layer_size=16,
        dropout_rate=0.0,
        ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.first_layer_size = first_layer_size
        self.dropout_rate = dropout_rate

        layer_sizes = [
            first_layer_size * 2 ** i for i in range(self.num_layers)
            ]
        
        self.in_conv = ConvBlock(
            in_channels, 
            layer_sizes[0],
            dropout_rate=dropout_rate,
        )

        encoder = []
        for i in range(self.num_layers - 1):
            in_ch = layer_sizes[i]
            out_ch = layer_sizes[i+1]
            encoder.append(
                DownBlock(in_ch, out_ch, dropout_rate=dropout_rate)
            )
        self.encoder = nn.ModuleList(encoder)
        
        decoder = []
        for i in range(self.num_layers - 1):
            in_ch = layer_sizes[-i-1]
            out_ch = layer_sizes[-i-2]
            decoder.append(
                UpBlock(in_ch, out_ch, dropout_rate=dropout_rate)
            )
        self.decoder = nn.ModuleList(decoder)

        self.out_conv = nn.Conv2d(
            layer_sizes[0], 
            out_channels, 
            kernel_size=1,
        )

    def forward(self, in_BCHW):
        skip_BCHW = []
        
        encode_BCHW = self.in_conv(in_BCHW)
        skip_BCHW.append(encode_BCHW)

        for layer in self.encoder:
            encode_BCHW = layer(encode_BCHW)
            skip_BCHW.append(encode_BCHW)
        
        decode_BCHW = encode_BCHW
        
        for i, layer in enumerate(self.decoder):
            skip_idx = len(skip_BCHW) - i - 2
            decode_BCHW = layer(
                decode_BCHW,
                skip_BCHW[skip_idx],
            )

        logits = self.out_conv(decode_BCHW)
        
        return (logits,)

if __name__ == '__main__':
    # python -m autotslabel.models.unet
    import torch
    from torchinfo import summary
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNet(in_channels=1, out_channels=2, num_layers=5, first_layer_size=16, dropout_rate=0.2)
    input_size = (2, 1, 513, 516)
    dtype = torch.float32
    summary(model, input_size=input_size, dtypes=[dtype], device=device)
    with torch.no_grad():
        output = model(torch.randn(input_size).to(device))
        print(output.shape)