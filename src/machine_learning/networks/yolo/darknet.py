from __future__ import annotations
from typing import Sequence

import torch
import torch.nn as nn

from machine_learning.networks import BaseNet


class DarkNet53(BaseNet):
    def __init__(
        self,
        default_img_shape: Sequence[int],
        num_anchors: int,
        num_classes: int,
    ):
        """yolo_v3 backbone network

        Args:
            default_img_shape (Sequence[int]): the shape of input image.
        """
        super().__init__()
        self.default_img_shape = default_img_shape  # (3, 416, 416) use to show the structre of the net
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.channels = (5 + self.num_classes) * self.num_anchors

        # Define the network backbone layers
        self.backbone_layers = nn.ModuleList(
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

        # fpn
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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        skips = []
        for i, layer in enumerate(self.backbone_layers):
            x = layer(x)
            if i in [6, 15, 24]:
                skips.append(x)

        # ----- The first layer of detection -----
        x3 = self.fpn.conv_x3(skips.pop())
        fmap3 = self.fpn.head_conv3(x3)
        x3_up = self.fpn.upsample(x3)

        # ----- The second layer of detection -----
        x2 = torch.cat([x3_up, self.fpn.conv_x2(skips.pop())], dim=1)
        fmap2 = self.fpn.head_conv2(x2)
        x2_up = self.fpn.upsample(x2)

        # ----- The third layer of detection -----
        x1 = torch.cat([x2_up, self.fpn.conv_x1(skips.pop())], dim=1)
        fmap1 = self.fpn.head_conv1(x1)

        return fmap1, fmap2, fmap3

    def view_structure(self) -> None:
        from torchinfo import summary

        dummy_input = torch.randn(1, *self.default_img_shape, device=self.device)

        summary(self, input_data=dummy_input)
