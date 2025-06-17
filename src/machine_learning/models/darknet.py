from __future__ import annotations
from typing import Sequence

import torch
import torch.nn as nn

from machine_learning.models import BaseNet


class ConvBNLeaky(nn.Module):
    """Conv2D+BN+LeakyReLU"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block (containing two convolutional layers)"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBNLeaky(channels, channels * 2, 1)
        self.conv2 = ConvBNLeaky(channels * 2, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        return out + residual


class Darknet(BaseNet):
    def __init__(self, input_size: Sequence[int]):
        """yolo_v3 backbone network

        Args:
            input_size (Sequence[int]): the size of input image.
        """
        super().__init__()
        self.input_size = input_size  # (3, 416, 416)

        # Define the network layer
        self.layers = nn.ModuleList(
            [
                # Initial Downsampling
                ConvBNLeaky(3, 32, 3, padding=1),  # (32, 416, 416)
                ConvBNLeaky(32, 64, 3, stride=2, padding=1),  # Downsampling (64, 208, 208)
                # Stage 1
                ResidualBlock(64),  # (64, 208, 208)
                ConvBNLeaky(64, 128, 3, stride=2, padding=1),  # Downsampling (128, 104, 104)
                # Stage 2
                *[ResidualBlock(128) for _ in range(2)],  # (128, 104, 104)
                ConvBNLeaky(128, 256, 3, stride=2, padding=1),  # Downsampling (256, 52, 52)
                # Stage 3
                *[ResidualBlock(256) for _ in range(8)],  # (256, 52, 52)
                ConvBNLeaky(256, 512, 3, stride=2, padding=1),  # Downsampling (512, 26, 26)
                # Stage 4
                *[ResidualBlock(512) for _ in range(8)],  # (512, 26, 26)
                ConvBNLeaky(512, 1024, 3, stride=2, padding=1),  # Downsampling (1024, 13, 13)
                # Stage 5
                *[ResidualBlock(1024) for _ in range(4)],
            ]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        skips = []

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in [6, 15, 24]:
                skips.append(x)

        deep_feature = skips.pop()
        mid_feature = skips.pop()
        shallow_feature = skips.pop()

        return shallow_feature, mid_feature, deep_feature

    def view_structure(self) -> None:
        from torchinfo import summary

        dummy_input = torch.randn(1, *self.input_size, device=self.device)

        summary(self, input_data=dummy_input)


class FPN(BaseNet):
    def __init__(self, num_anchors: int, num_classes: int) -> None:
        """Feature Pyramid network

        Args:
            skips (list[torch.Tensor]): skips output from the darknet backbone network.
        """
        super().__init__()

        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.channels = (5 + self.num_classes) * self.num_anchors

        self.fpn = nn.ModuleDict(
            {
                # convolutional layers for feature fusion
                "conv_x1": ConvBNLeaky(256, 128, 1),
                "conv_x2": ConvBNLeaky(512, 256, 1),
                "conv_x3": ConvBNLeaky(1024, 512, 1),
                # upsampling layer
                "upsample": nn.Upsample(scale_factor=2, mode="nearest"),
                # Detection head (The last layer does not use the activation function and maintains the logits output)
                "head_conv1": nn.Sequential(ConvBNLeaky(896, 448, 3, padding=1), nn.Conv2d(448, self.channels, 1)),
                "head_conv2": nn.Sequential(ConvBNLeaky(768, 384, 3, padding=1), nn.Conv2d(384, self.channels, 1)),
                "head_conv3": nn.Sequential(ConvBNLeaky(512, 256, 3, padding=1), nn.Conv2d(256, self.channels, 1)),
            }
        )

    def forward(
        self, shallow_feature: torch.Tensor, mid_feature: torch.Tensor, deep_feature: torch.Tensor
    ) -> tuple[torch.Tensor]:
        """Feature pyramid network forward propagation

        Args:
            skips (list[torch.Tensor]): the dim of skips : [(256, 52, 52),(512, 26, 26),(1024, 13, 13)].

        Returns:
            torch.Tensor: the fimg output.
        """
        # skips: [(256,52,52),(512,26,26),(1024,13,13)]

        # ----- The first layer of detection -----
        x3 = self.fpn.conv_x3(deep_feature)  # (1024,13,13) -> (512,13,13)
        detection3 = self.fpn.head_conv3(x3)  # (512,13,13) -> (255,13,13)
        x3_up = self.fpn.upsample(x3)  # (512,13,13) -> (512,26,26)

        # ----- The second layer of detection -----
        x2 = torch.cat([x3_up, self.fpn.conv_x2(mid_feature)], dim=1)  # (512,26,26) cat (256,26,26) -> (768,26,26)
        detection2 = self.fpn.head_conv2(x2)  # (768,26,26) -> (255,26,26)
        x2_up = self.fpn.upsample(x2)  # (768,26,26) -> (768,52,52)

        # ----- The third layer of detection -----
        x1 = torch.cat([x2_up, self.fpn.conv_x1(shallow_feature)], dim=1)  # (768,52,52) cat (128,52,52) -> (896,52,52)
        detection1 = self.fpn.head_conv1(x1)  # (896,52,52) -> (255,13,13)

        return detection1, detection2, detection3  # 52x52, 26x26, 13x13

    def view_structure(self):
        from torchinfo import summary

        # When the network forward function has multiple parameter inputs, create a virtual data pass that conforms to
        # the input structure
        dummy_input = [
            torch.randn(1, 256, 52, 52, device=self.device),  # corresponding to the shallow feature map
            torch.randn(1, 512, 26, 26, device=self.device),  # corresponding to the middle feature map
            torch.randn(1, 1024, 13, 13, device=self.device),  # # Corresponding to the deep feature map
        ]

        summary(self, input_data=dummy_input)
