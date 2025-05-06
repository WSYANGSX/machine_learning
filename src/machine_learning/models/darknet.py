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
    """残差块（包含两个卷积层）"""

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
        """yolo_v3主干网络

        Args:
            input_size (Sequence[int]): 输入图像尺寸大小.
        """
        super().__init__()
        self.input_size = input_size  # (3, 416, 416)

        # 定义网络层
        self.layers = nn.ModuleList(
            [
                # 初始下采样
                ConvBNLeaky(3, 32, 3, padding=1),  # (32, 416, 416)
                ConvBNLeaky(32, 64, 3, stride=2, padding=1),  # 下采样 (64, 208, 208)
                # Stage 1
                ResidualBlock(64),  # (64, 208, 208)
                ConvBNLeaky(64, 128, 3, stride=2, padding=1),  # 下采样 (128, 104, 104)
                # Stage 2
                *[ResidualBlock(128) for _ in range(2)],  # (128, 104, 104)
                ConvBNLeaky(128, 256, 3, stride=2, padding=1),  # 下采样 (256, 52, 52)
                # Stage 3
                *[ResidualBlock(256) for _ in range(8)],  # (256, 52, 52)
                ConvBNLeaky(256, 512, 3, stride=2, padding=1),  # 下采样 (512, 26, 26)
                # Stage 4
                *[ResidualBlock(512) for _ in range(8)],  # (512, 26, 26)
                ConvBNLeaky(512, 1024, 3, stride=2, padding=1),  # 下采样 (1024, 13, 13)
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

        summary(self, (1, *self.input_size))  # 假设输入为416x416


class FPN(BaseNet):
    def __init__(self, num_anchors: int, num_classes: int) -> None:
        """特征金字塔网络

        Args:
            skips (list[torch.Tensor]): 来自darknet主干网络的跳跃链接.
        """
        super().__init__()

        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.channels = (5 + self.num_classes) * self.num_anchors

        # 特征金字塔网络（FPN）
        self.fpn = nn.ModuleDict(
            {
                # 用于特征融合的卷积层
                "conv_x1": ConvBNLeaky(256, 128, 1),
                "conv_x2": ConvBNLeaky(512, 256, 1),
                "conv_x3": ConvBNLeaky(1024, 512, 1),
                # 上采样层
                "upsample": nn.Upsample(scale_factor=2, mode="nearest"),
                # 检测头前的卷积
                "head_conv1": nn.Sequential(ConvBNLeaky(896, 448, 3, padding=1), ConvBNLeaky(448, self.channels, 1)),
                "head_conv2": nn.Sequential(ConvBNLeaky(768, 384, 3, padding=1), ConvBNLeaky(384, self.channels, 1)),
                "head_conv3": nn.Sequential(ConvBNLeaky(512, 256, 3, padding=1), ConvBNLeaky(256, self.channels, 1)),
            }
        )

    def forward(
        self, shallow_feature: torch.Tensor, mid_feature: torch.Tensor, deep_feature: torch.Tensor
    ) -> tuple[torch.Tensor]:
        """特征金字塔网络前向传播

        Args:
            skips (list[torch.Tensor]): skips维度: [(256, 52, 52),(512, 26, 26),(1024, 13, 13)].

        Returns:
            torch.Tensor: 检测输出信息.
        """
        # 特征金字塔网络
        # skips: [(256,52,52),(512,26,26),(1024,13,13)]

        # ----- 第一层检测 -----
        x3 = self.fpn.conv_x3(deep_feature)  # (1024,13,13) -> (512,13,13)
        detection3 = self.fpn.head_conv3(x3)  # (512,13,13) -> (255,13,13)
        x3_up = self.fpn.upsample(x3)  # (512,13,13) -> (512,26,26)

        # ----- 第二层检测 -----
        x2 = torch.cat([x3_up, self.fpn.conv_x2(mid_feature)], dim=1)  # (512,26,26) cat (256,26,26) -> (768,26,26)
        detection2 = self.fpn.head_conv2(x2)  # (768,26,26) -> (255,26,26)
        x2_up = self.fpn.upsample(x2)  # (768,26,26) -> (768,52,52)

        # ----- 第三层检测 -----
        x1 = torch.cat([x2_up, self.fpn.conv_x1(shallow_feature)], dim=1)  # (768,52,52) cat (128,52,52) -> (896,52,52)
        detection1 = self.fpn.head_conv1(x1)  # (896,52,52) -> (255,13,13)

        return detection1, detection2, detection3  # 对应52x52, 26x26, 13x13三个尺度

    def view_structure(self):
        from torchinfo import summary

        # 创建符合输入结构的虚拟数据
        dummy_input = [
            torch.randn(1, 256, 52, 52),  # 对应浅层特征图
            torch.randn(1, 512, 26, 26),  # 中层特征图
            torch.randn(1, 1024, 13, 13),  # 深层特征图
        ]

        # 正确调用方式
        summary(self, input_data=dummy_input)


if __name__ == "__main__":
    fpn = FPN(3, 80)
    fpn.view_structure()
    dummy_input = [
        torch.randn(1, 256, 52, 52),  # 对应浅层特征图
        torch.randn(1, 512, 26, 26),  # 中层特征图
        torch.randn(1, 1024, 13, 13),  # 深层特征图
    ]
    det1, det2, det3 = fpn(*dummy_input)
    print(det1.shape, det2.shape, det3.shape)
