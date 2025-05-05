from __future__ import annotations
from typing import Sequence

import torch
import torch.nn as nn

from machine_learning.models import BaseNet  # 假设基类已定义基础功能


class ConvBNLeaky(nn.Module):
    """卷积+BN+LeakyReLU基本单元"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """残差块（包含两个卷积层）"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBNLeaky(channels, channels // 2, 1)
        self.conv2 = ConvBNLeaky(channels // 2, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        return out + residual  # 残差连接


class Darknet(BaseNet):
    def __init__(self, input_size: Sequence[int]):
        super().__init__()
        self.input_size = input_size  # (3, 416, 416)

        # 定义网络层
        self.layers = nn.ModuleList(
            [
                # 初始下采样
                ConvBNLeaky(3, 32, 3, padding=1),  # (32, 416, 416)
                ConvBNLeaky(32, 64, 3, stride=2, padding=1),  # 下采样 (64, 209, 209)
                # Stage 1
                ResidualBlock(64),  # (64, 209, 209)
                ConvBNLeaky(64, 128, 3, stride=2, padding=1),  # 下采样 (128, 105, 105)
                # Stage 2
                *[ResidualBlock(128) for _ in range(2)],  # (128, 105, 105)
                ConvBNLeaky(128, 256, 3, stride=2, padding=1),  # 下采样 (256, 53, 53)
                # Stage 3
                *[ResidualBlock(256) for _ in range(8)],  # (256, 53, 53)
                ConvBNLeaky(256, 512, 3, stride=2, padding=1),  # 下采样 (512, 53, 53)
                # Stage 4
                *[ResidualBlock(512) for _ in range(8)],
                ConvBNLeaky(512, 1024, 3, stride=2, padding=1),  # 下采样
                # Stage 5
                *[ResidualBlock(1024) for _ in range(4)],
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []  # 保存三个尺度的输出（用于YOLO多尺度检测）

        for i, layer in enumerate(self.layers):
            x = layer(x)
            # 记录三个下采样点的输出（对应不同尺度特征图）
            if i in [5, 13, 20]:  # 根据网络结构设定的特征图位置
                outputs.append(x)

        return outputs[-3:]  # 返回最后三个尺度的特征图

    def view_structure(self) -> None:
        """可视化网络结构"""
        from torchinfo import summary

        summary(self, (1, *self.input_size))  # 假设输入为416x416
