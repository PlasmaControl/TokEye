import torch
import torch.nn as nn
import torch.nn.functional as F


class crop(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        N, C, H, W = x.shape
        x = x[0:N, 0:C, 0 : H - 1, 0:W]  # crop last row
        return x


class shift(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift_down = nn.ZeroPad2d((0, 0, 1, 0))
        self.crop = crop()

    def forward(self, x):
        x = self.shift_down(x)
        x = self.crop(x)
        return x


class super_shift(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, hole_size=1):
        shift_offset = (hole_size + 1) // 2  # hole_size must be odd

        x = nn.ZeroPad2d((0, 0, shift_offset, 0))(x)  # left right top bottom
        N, C, H, W = x.shape
        x = x[0:N, 0:C, 0 : H - shift_offset, 0:W]  # crop last rows
        return x


class Conv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bias=True,
        blind=True,
        stride=1,
        padding=1,
        kernel_size=3,
    ):
        super().__init__()
        self.blind = blind
        if blind:
            self.shift_down = nn.ZeroPad2d((0, 0, 1, 0))  # left right top bottom
            self.crop = crop()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        if self.blind:
            x = self.shift_down(x)
        x = self.conv(x)
        if self.blind:
            x = self.crop(x)
        x = self.relu(x)
        return x


class Pool(nn.Module):
    def __init__(self, blind=True):
        super().__init__()
        self.blind = blind
        if blind:
            self.shift = shift()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        if self.blind:
            x = self.shift(x)
        x = self.pool(x)
        return x


class rotate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x90 = x.transpose(2, 3).flip(3)
        x180 = x.flip(2).flip(3)
        x270 = x.transpose(2, 3).flip(2)
        x = torch.cat((x, x90, x180, x270), dim=0)
        return x


class unrotate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x0, x90, x180, x270 = torch.chunk(x, 4, dim=0)
        x90 = x90.transpose(2, 3).flip(2)
        x180 = x180.flip(2).flip(3)
        x270 = x270.transpose(2, 3).flip(3)
        x = torch.cat((x0, x90, x180, x270), dim=1)
        return x


class ENC_Conv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bias=True,
        reduce=True,
        blind=True,
    ):
        super().__init__()
        self.reduce = reduce
        self.conv1 = Conv(in_channels, out_channels, bias=bias, blind=blind)
        if reduce:
            self.pool = Pool(blind=blind)

    def forward(self, x):
        x = self.conv1(x)
        if self.reduce:
            x = self.pool(x)
        return x


class DEC_Conv(nn.Module):
    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        bias=True,
        increase=True,
        blind=True,
    ):
        super().__init__()
        self.increase = increase
        self.conv1 = Conv(in_channels, mid_channels, bias=bias, blind=blind)
        self.conv2 = Conv(mid_channels, out_channels, bias=bias, blind=blind)
        if increase:
            self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x, x_in):
        x = torch.cat((x, x_in), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        if self.increase:
            x = self.upsample(x)
        return x


class Blind_UNet(nn.Module):
    def __init__(
        self,
        n_channels=3,
        n_layers=3,
        mid_channels=48,
        n_output=96,
        bias=True,
        blind=True,
    ):
        super().__init__()

        # input
        self.encoder_input = Conv(
            in_channels=n_channels,
            out_channels=mid_channels,
            bias=bias,
            blind=blind,
        )

        # encoder
        layers = []
        for _ in range(n_layers):
            layers.append(
                ENC_Conv(
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    bias=bias,
                    blind=blind,
                )
            )
        self.encoder = nn.ModuleList(layers)

        # bottleneck
        layers = []
        layers.append(
            ENC_Conv(
                in_channels=mid_channels,
                out_channels=mid_channels,
                bias=bias,
                blind=blind,
            )
        )
        layers.append(
            ENC_Conv(
                in_channels=mid_channels,
                out_channels=mid_channels,
                bias=bias,
                reduce=False,
                blind=blind,
            )
        )
        layers.append(
            nn.Upsample(
                scale_factor=2,
                mode="nearest",
            )
        )
        self.bottleneck = nn.Sequential(*layers)

        # decoder
        layers = []
        layers.append(
            DEC_Conv(
                in_channels=mid_channels * 2,
                mid_channels=mid_channels * 2,
                out_channels=mid_channels * 2,
                bias=bias,
                blind=blind,
            )
        )
        for _ in range(n_layers - 1):
            layers.append(
                DEC_Conv(
                    in_channels=mid_channels * 3,
                    mid_channels=mid_channels * 2,
                    out_channels=mid_channels * 2,
                    bias=bias,
                    blind=blind,
                )
            )
        self.decoder = nn.ModuleList(layers)

        # output
        self.decoder_output = DEC_Conv(
            in_channels=mid_channels * 2 + n_channels,
            mid_channels=mid_channels * 2,
            out_channels=n_output,
            bias=bias,
            increase=False,
            blind=blind,
        )

    def forward(self, input_BCHW):
        x = self.encoder_input(input_BCHW)

        skip_BCHW = []
        for layer in self.encoder:
            x = layer(x)
            skip_BCHW.append(x)

        x = self.bottleneck(x)

        for i, layer in enumerate(self.decoder):
            skip_idx = len(skip_BCHW) - i - 1
            x = layer(x, skip_BCHW[skip_idx])

        # x = self.decoder_output(x, input_BCHW)
        return x


class ATBSN(nn.Module):
    def __init__(
        self,
        n_channels=3,
        n_layers=3,
        unet_channels=48,
        concat_channels=96,
        n_output=3,
        bias=True,
        blind=True,
    ):
        super().__init__()
        self.blind = blind
        self.rotate = rotate()
        self.unet = Blind_UNet(
            n_channels=n_channels,
            n_layers=n_layers,
            mid_channels=unet_channels,
            n_output=concat_channels,
            bias=bias,
            blind=blind,
        )
        self.shift = super_shift()
        self.unrotate = unrotate()

        n_rotations = 4

        # undo the rotations
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels=concat_channels * n_rotations,
                out_channels=concat_channels * n_rotations,
                kernel_size=1,
                bias=bias,
            )
        )
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        layers.append(
            nn.Conv2d(
                in_channels=concat_channels * n_rotations,
                out_channels=concat_channels,
                kernel_size=1,
                bias=bias,
            )
        )
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        layers.append(
            nn.Conv2d(
                in_channels=concat_channels,
                out_channels=n_output,
                kernel_size=1,
                bias=bias,
            )
        )
        self.unconcat = nn.Sequential(*layers)

        with torch.no_grad():
            self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, a=0.1)
                m.bias.data.zero_()
        nn.init.kaiming_normal_(
            self.unconcat[-1].weight.data,
            nonlinearity="linear",
        )

    def forward(self, x, hole_size=1):
        x = self.rotate(x)
        x = self.unet(x)

        if self.blind:
            x = self.shift(x, hole_size)

        x = self.unrotate(x)
        x = self.unconcat(x)

        return x


class N_BSN(nn.Module):  # student c, 1.00m (1.02m in the paper is a typo)
    def __init__(
        self,
        n_channels=3,
        n_layers=3,
        mid_channels=48,
        n_output=3,
        bias=True,
        blind=False,
    ):
        super().__init__()
        self.unet = Blind_UNet(
            n_channels=n_channels,
            n_layers=n_layers,
            mid_channels=mid_channels,
            n_output=n_output,
            bias=bias,
            blind=blind,
        )

        with torch.no_grad():
            self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, a=0.1)
                m.bias.data.zero_()

    def forward(self, x):
        x0 = self.unet(x)

        return x0


class DoubleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.nbsn = N_BSN()
        self.bsn = ATBSN()
        # freeze bsn
        for param in self.bsn.parameters():
            param.requires_grad = False
        self.bsn.eval()

    def forward(
        self,
        x,
        hole_size=[0, 1, 3, 5, 7, 9, 11],
        mode="test",
    ):
        if mode == "train":
            res_list = []
            with torch.no_grad():
                for hs in hole_size:
                    res_list.append(self.bsn(x, hs))
            x_atbsn = self.nbsn(x)
            return x_atbsn, res_list
        elif mode == "test":
            x_atbsn = self.nbsn(x)
            return x_atbsn


if __name__ == "__main__":
    # python -m autotslabel.autosegment.preprocessing.bsn
    from torchinfo import summary

    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_channels = 1

    input_size = (2, n_channels, 512, 512)  # B, C, H, W

    # model = N_BSN(n_channels=3, n_output=3, bias=True, blind=False)
    model = ATBSN(
        n_channels=n_channels,
        n_layers=2,
        unet_channels=48,
        concat_channels=96,
        n_output=n_channels,
        bias=True,
        blind=True,
    ).to(device)

    summary(model, input_size=input_size)

    test_input = torch.randn(input_size).to(device)
    with torch.no_grad():
        output = model(test_input, hole_size=2)
    print(f"input shape: {test_input.shape}")
    print(f"output shape: {output.shape}")
