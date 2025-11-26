import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout_rate=0.0):
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
        
        layers.extend([
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        ])
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(p=dropout_rate))
        
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_rate=dropout_rate)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super().__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConvBlock(in_channels + out_channels, out_channels, dropout_rate=dropout_rate)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, 
            [diffX // 2, diffX - diffX // 2,
            diffY // 2, diffY - diffY // 2],
        )
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

if __name__ == "__main__":
    # python -m autotslabel.models.modules.nn
    model = ConvBlock(3, 64)
    x = torch.randn(1, 3, 256, 256)
    print(model(x).shape)

    model = DownBlock(3, 64)
    x = torch.randn(1, 3, 256, 256)
    print(model(x).shape)

    model = UpBlock(3, 64)
    x1 = torch.randn(1, 3, 128, 128)
    x2 = torch.randn(1, 3, 256, 256)
    print(model(x1, x2).shape)