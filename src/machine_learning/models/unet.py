import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from machine_learning.models import BaseNet


# 时间步嵌入层
class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim * 4)
        self.activate = nn.SiLU()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)  # 频率衰减系数
        emb = torch.exp(-torch.arange(half_dim, device=t.device) * emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        emb = self.linear1(emb)
        emb = self.activate(emb)
        emb = self.linear2(emb)
        return emb


# 残差块（包含时间步嵌入）
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)  # 图像大小不变
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.time_proj = nn.Linear(time_dim, out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)  # 图像大小不变
        self.norm2 = nn.GroupNorm(32, out_channels)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.norm1(h)

        # 添加时间步信息
        t_emb = self.time_proj(self.act(t_emb))[:, :, None, None]
        h = h + t_emb

        h = self.act(h)
        h = self.conv2(self.dropout(h))
        h = self.norm2(h)

        return h + self.shortcut(x)


# 注意力模块
class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q = self.q(h).view(B, C, -1).permute(0, 2, 1)
        k = self.k(h).view(B, C, -1)
        v = self.v(h).view(B, C, -1)

        attn = torch.bmm(q, k) * (C**-0.5)
        attn = F.softmax(attn, dim=-1)

        h = torch.bmm(v, attn.permute(0, 2, 1))
        h = h.view(B, C, H, W)
        return x + self.proj_out(h)  # 在数据传播过程中保留原始信息并增强全局依赖


class UNet(BaseNet):
    def __init__(self, input_size, time_dim, in_channels, out_channels, down_channels, up_channels):
        super().__init__()

        self.input_size = input_size
        self.time_dim = time_dim
        self.in_channels = in_channels
        self.down_channels = down_channels
        self.up_channels = up_channels
        self.out_channels = out_channels

        self.time_embed = TimestepEmbedding(time_dim)

        # 下采样部分
        self.down_blocks = nn.ModuleList()
        in_ch = self.in_channels
        for out_ch in down_channels:
            block = nn.ModuleList(
                [
                    ResidualBlock(in_ch, out_ch, self.time_dim),
                    AttentionBlock(out_ch),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1),
                ]
            )
            self.down_blocks.append(block)
            in_ch = out_ch

        # 中间部分
        self.mid_block1 = ResidualBlock(in_ch, in_ch, self.time_dim)
        self.mid_attn = AttentionBlock(in_ch)
        self.mid_block2 = ResidualBlock(in_ch, in_ch, self.time_dim)

        # 上采样部分
        self.up_blocks = nn.ModuleList()
        for idx, out_ch in enumerate(self.up_channels):
            block = nn.ModuleList(
                [
                    nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1),
                    ResidualBlock(out_ch * 2, out_ch, self.time_dim),
                    AttentionBlock(out_ch),
                ]
            )
            self.up_blocks.append(block)
            in_ch = out_ch

        # 输出层
        self.out_norm = nn.GroupNorm(32, in_ch)
        self.out_conv = nn.Conv2d(in_ch, self.out_channels, kernel_size=3, padding=1)

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
