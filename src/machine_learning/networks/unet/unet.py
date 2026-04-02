from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from machine_learning.networks.base import BaseNet


class DConvBNAct(nn.Module):
    """(Conv => BN => Act) * 2."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: Literal["relu", "gelu", "leaky_relu"] = "relu",
    ):
        super().__init__()
        if activation == "relu":
            act_layer = nn.ReLU(inplace=True)
        elif activation == "gelu":
            act_layer = nn.GELU()
        elif activation == "leaky_relu":
            act_layer = nn.LeakyReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            act_layer,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            act_layer,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DownBlock(nn.Module):
    """Downscaling with maxpool then double conv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: Literal["relu", "gelu", "leaky_relu"] = "relu",
    ):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DConvBNAct(in_channels, out_channels, activation=activation))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upscaling then double conv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = True,
        activation: Literal["relu", "gelu", "leaky_relu"] = "relu",
    ):
        super().__init__()
        self.activation = activation

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DConvBNAct(in_channels, out_channels, activation=activation)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DConvBNAct(in_channels, out_channels, activation=activation)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        # input is CHW, pad if the sizes don't perfectly match
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        if diffY > 0 or diffX > 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # Concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(BaseNet):
    """
    Standard U-Net implementation.
    Features modern 'same' padding to prevent cropping and simplify skip connections.
    """

    def __init__(
        self,
        imgsz: int = 256,
        in_channels: int = 3,
        num_classes: int = 1,
        bilinear: bool = False,
        activation: Literal["relu", "gelu", "leaky_relu"] = "relu",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.imgsz = imgsz
        self.activation = activation

        factor = 2 if bilinear else 1

        # Encoder
        self.inc = DConvBNAct(in_channels, 64, activation=activation)
        self.down1 = DownBlock(64, 128, activation=activation)
        self.down2 = DownBlock(128, 256, activation=activation)
        self.down3 = DownBlock(256, 512, activation=activation)
        self.down4 = DownBlock(512, 1024 // factor, activation=activation)

        # Decoder
        self.up1 = UpBlock(1024, 512 // factor, self.bilinear, activation=activation)
        self.up2 = UpBlock(512, 256 // factor, self.bilinear, activation=activation)
        self.up3 = UpBlock(256, 128 // factor, self.bilinear, activation=activation)
        self.up4 = UpBlock(128, 64, self.bilinear, activation=activation)

        # Final prediction head
        self.outc = nn.Conv2d(64, self.num_classes, kernel_size=1)

    @property
    def dummy_input(self) -> torch.Tensor:
        """
        Provides a dummy input for the BaseNet structure viewer and FLOPs calculator.
        """
        return torch.randn(1, self.in_channels, self.imgsz, self.imgsz, device=self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder pathway
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder pathway with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits


if __name__ == "__main__":
    model = UNet(imgsz=256, in_channels=3, num_classes=1, bilinear=False, activation="relu")
    model.view_structure()
