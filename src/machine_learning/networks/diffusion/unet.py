from typing import Sequence

import torch
import torch.nn as nn

from machine_learning.networks import BaseNet
from machine_learning.modules import AttentionBlock
from machine_learning.utils.layers import cal_conv_output_size, cal_convtrans_output_size


# 时间步嵌入层
class TimestepEmbeddingBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim * 4)
        self.act = nn.SiLU()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)  # 频率衰减系数
        emb = torch.exp(-torch.arange(half_dim, device=t.device) * emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        emb = self.linear1(emb)
        emb = self.act(emb)
        emb = self.linear2(emb)
        return emb


# 残差块（包含时间步嵌入）
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)  # 图像大小不变
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.time_proj = nn.Linear(time_dim * 4, out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)  # 图像大小不变
        self.norm2 = nn.GroupNorm(32, out_channels)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)

        # 添加时间步信息
        t_emb = self.time_proj(self.act(t_emb))[:, :, None, None]
        h = h + t_emb

        h = self.act(h)
        h = self.conv2(self.dropout(h))
        h = self.norm2(h)

        return h + self.shortcut(x)


class UNet(BaseNet):
    def __init__(
        self,
        input_size: Sequence[int],
        time_dim: int,
        in_channels: int,
        out_channels: int,
        down_channels: Sequence[int],
        up_channels: Sequence[int],
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = None

        self.time_dim = time_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.down_channels = down_channels
        self.up_channels = up_channels

        # 时间嵌入
        self.time_embed = TimestepEmbedding(time_dim)
        self.act = nn.SiLU()

        # 下采样部分
        self.down_blocks = nn.ModuleList()
        in_ch = self.in_channels
        in_size = self.input_size
        self.down_size = [self.input_size[1:]]
        for out_ch in down_channels:
            block = nn.ModuleList(
                [
                    ResidualBlock(in_ch, out_ch, self.time_dim),  # 不改变图像大小,改变通道数
                    AttentionBlock(out_ch),  # 不改变图像大小
                    nn.Conv2d(
                        out_ch,
                        out_ch,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),  # 下采样，改变图像大小
                ]
            )
            self.output_size = cal_conv_output_size((out_ch, *in_size[1:]), block[2])
            self.down_size.append(self.output_size[1:])
            self.down_blocks.append(block)

            in_ch = out_ch
            in_size = self.output_size

        # 中间部分，不改变大小和通道数
        self.mid_block1 = ResidualBlock(in_ch, in_ch, self.time_dim)
        self.mid_attn = AttentionBlock(in_ch)
        self.mid_block2 = ResidualBlock(in_ch, in_ch, self.time_dim)

        # 上采样部分
        self.up_blocks = nn.ModuleList()
        for idx, out_ch in enumerate(self.up_channels):
            block = nn.ModuleList(
                [
                    nn.ConvTranspose2d(
                        in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1
                    ),  # 上采样，改变图像大小和通道数
                    ResidualBlock(out_ch * 2, out_ch, self.time_dim),
                    AttentionBlock(out_ch),
                ]
            )
            self.output_size = cal_convtrans_output_size(in_size, block[0])

            if self.output_size[1:] != self.down_size[-(2 + idx)]:
                block = nn.ModuleList(
                    [
                        nn.ConvTranspose2d(
                            in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=0
                        ),  # 上采样，改变图像大小和通道数
                        ResidualBlock(out_ch * 2, out_ch, self.time_dim),
                        AttentionBlock(out_ch),
                    ]
                )
                self.output_size = cal_convtrans_output_size(in_size, block[0])

            self.up_blocks.append(block)

            in_ch = out_ch
            in_size = self.output_size

        # 输出层
        self.out_norm = nn.GroupNorm(32, in_ch)
        self.out_conv = nn.Conv2d(in_ch, self.out_channels, kernel_size=3, padding=1)

        self.output_size = cal_conv_output_size(in_size, self.out_conv)

    def forward(self, x, t):
        # 时间嵌入
        t_emb = self.time_embed(t)

        # 存储跳跃连接
        skips = []

        # 下采样过程
        for block, attn, downsample in self.down_blocks:
            x = block(x, t_emb)
            x = attn(x)
            skips.append(x)
            x = downsample(x)

        # 中间处理
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)

        # 上采样过程
        for upsample, block, attn in self.up_blocks:
            x = upsample(x)
            x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, t_emb)
            x = attn(x)

        # 输出
        x = self.out_norm(x)
        x = self.act(x)
        return self.out_conv(x)

    def view_structure(self):
        pass
