from typing import Sequence
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from mamba_ssm.modules.mamba_simple import Mamba
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import DSC3k, DSBottleneck, C3AH, HyperACE, GhostConv


class AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        h = self.norm(x)
        q = self.q(h).view(B, C, -1).permute(0, 2, 1)
        k = self.k(h).view(B, C, -1)
        v = self.v(h).view(B, C, -1)

        attn = torch.bmm(q, k) * (C**-0.5)
        attn = F.softmax(attn, dim=-1)

        h = torch.bmm(v, attn.permute(0, 2, 1))
        h = h.view(B, C, H, W)
        return x + self.proj_out(h)  # Retain the original information and enhance global dependencies


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self._block = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1
            ),  # 图像大小不变
            nn.GroupNorm(32, out_channels),
            nn.ReLU(),
            nn.Dropout1d(dropout),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1),  # 图像大小不变
            nn.GroupNorm(32, out_channels),
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        return self.shortcut(x) + self._block(x)


class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self._block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.GroupNorm(32, out_channels),
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        return self.shortcut(x) + self._block(x)


class ConvBNLeakyBlock(nn.Module):
    """Conv2D+BN+LeakyReLU"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBNLeakyBlock(channels, channels, 1)
        self.conv2 = ConvBNLeakyBlock(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x)) + x


class CFuseModule(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.conv_out = Conv(2 * c_in, c_in, 1)

    def forward(self, x):
        x_cat = torch.cat([x[0], x[1]], dim=1)
        out = self.conv_out(x_cat)
        return out


class CMCA(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out

        self.conv1 = nn.Conv2d(c_in, c_out, 1)
        self.conv2 = nn.Conv2d(c_in, c_out, 1)
        self.act1 = nn.Tanh()

        self.conv3 = nn.Conv2d(2 * c_out, 1, 1)
        self.act2 = nn.Sigmoid()

        # channel attention
        self.c_i1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Conv2d(c_out, c_out // 8, 1),
            nn.GELU(),
            nn.Conv2d(c_out // 8, c_out, 1),
            nn.Sigmoid(),
        )
        self.c_i2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Conv2d(c_out, c_out // 8, 1),
            nn.GELU(),
            nn.Conv2d(c_out // 8, c_out, 1),
            nn.Sigmoid(),
        )

    def forward(self, X):
        x1 = self.act1(self.conv1(X[0]))
        x2 = self.act1(self.conv2(X[1]))

        z = torch.cat([x1, x2], dim=1)
        z = self.act2(self.conv3(z))

        x1_m = z * x1
        x2_m = z * x2

        x1_out = x1_m + x1_m * self.c_i1(x2_m)
        x2_out = x2_m + x2_m * self.c_i2(x1_m)

        return x1_out, x2_out


class CHyperACE(nn.Module):
    def __init__(
        self,
        c1,
        c2,
        n=1,
        num_hyperedges=8,
        dsc3k=True,
        shortcut=False,
        e1=0.5,
        e2=1,
        context="both",
    ):
        super().__init__()
        self.c = int(c2 * e1)
        self.cv1 = Conv(c1, 3 * self.c, 1, 1)
        self.cv2 = Conv((4 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            DSC3k(self.c, self.c, 2, shortcut, k1=3, k2=7) if dsc3k else DSBottleneck(self.c, self.c, shortcut=shortcut)
            for _ in range(n)
        )
        self.cmca = CMCA(c1, c1)
        self.fuse = CFuseModule(c1)
        self.branch1 = C3AH(self.c, self.c, e2, num_hyperedges, context)
        self.branch2 = C3AH(self.c, self.c, e2, num_hyperedges, context)

    def forward(self, X):
        X = self.cmca(X)
        x = self.fuse(X)
        y = list(self.cv1(x).chunk(3, 1))
        out1 = self.branch1(y[1])
        out2 = self.branch2(y[1])
        y.extend(m(y[-1]) for m in self.m)
        y[1] = out1
        y.append(out2)

        return self.cv2(torch.cat(y, 1))


class CHyperACEV2(nn.Module):
    def __init__(
        self,
        c1,
        c2,
        n=1,
        num_hyperedges=8,
        dsc3k=True,
        shortcut=False,
        e1=0.5,
        e2=1,
        context="both",
    ):
        super().__init__()
        self.c = int(c2 * e1)
        self.cv1 = Conv(c1, 3 * self.c, 1, 1)
        self.cv2 = Conv(c1, self.c, 1, 1)
        self.cv3 = Conv((5 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            DSC3k(self.c, self.c, 2, shortcut, k1=3, k2=7) if dsc3k else DSBottleneck(self.c, self.c, shortcut=shortcut)
            for _ in range(n)
        )
        self.cmca = CMCA(c1, c1)
        self.fuse1 = CFuseModule(c1)
        self.fuse2 = CFuseModule(c1)
        self.branch1 = C3AH(self.c, self.c, e2, num_hyperedges, context)
        self.branch2 = C3AH(self.c, self.c, e2, num_hyperedges, context)

    def forward(self, X):
        x = self.fuse2(X)
        y = list(self.cv1(x).chunk(3, 1))
        out1 = self.branch1(y[1])
        out2 = self.branch2(y[1])
        y.extend(m(y[-1]) for m in self.m)
        y[1] = out1
        y.append(out2)

        X_hat = self.cmca(X)
        out3 = self.cv2(self.fuse1(X_hat))
        y.append(out3)

        return self.cv3(torch.cat(y, 1))


class MMFullPAD_Tunnel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate1 = nn.Parameter(torch.tensor(0.0))
        self.gate2 = nn.Parameter(torch.tensor(0.0))
        self.gate3 = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        out = x[0] + self.gate1 * x[1] + self.gate2 * x[2] + self.gate3 * x[3]
        return out


class MFD(nn.Module):
    """Multimodal feature random shielding module"""

    def __init__(self, p=0.15):
        super().__init__()
        self.p = p
        self.alpha = 1.0 / (1.0 - 0.5 * self.p) if self.p > 0 else 1.0

    def forward(self, X: Sequence[list[torch.Tensor]]):
        if not self.training or self.p == 0:
            return X

        if torch.rand(1).item() < self.p:
            drop_idx = random.randint(0, 1)
            keep_idx = 1 - drop_idx

            for i in range(len(X[drop_idx])):
                X[drop_idx][i] = torch.zeros_like(X[drop_idx][i])

            for i in range(len(X[keep_idx])):
                X[keep_idx][i] = X[keep_idx][i] * self.alpha
        else:
            for modality in X:
                for i in range(len(modality)):
                    modality[i] = modality[i] * self.alpha

        return X


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True)
        return x / (norm / (x.shape[-1] ** 0.5) + self.eps) * self.weight


class ECABlock(nn.Module):
    def __init__(self, ch, k_size=3):
        super().__init__()
        self.ch = ch
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 1D conv expects (B, C=1, L), we keep it as in typical ECA impl
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        y = self.avg_pool(x)  # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)  # (B, 1, C)
        y = self.conv(y)  # (B, 1, C)
        y = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1))  # (B, C, 1, 1)
        return x * y.expand_as(x)


class LowRankFeedForward(nn.Module):
    def __init__(self, dim, rank_ratio=0.5, dropout=0.05):
        super().__init__()
        hidden_dim = max(1, int(dim * rank_ratio))
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.dwconv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x.transpose(1, 2)).transpose(1, 2)
        x = self.act(x)
        x = self.fc2(x)
        return self.dropout(x)


class SingleMambaBlock(nn.Module):
    def __init__(self, dim, H, W, shared_block=None, scale_init=1.0):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.block = shared_block or Mamba(
            dim, expand=1, d_state=8, bimamba_type="v6", if_devide_out=True, use_norm=True, input_h=H, input_w=W
        )
        self.gamma = nn.Parameter(scale_init * torch.ones(dim))

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.block(x)
        return residual + x * self.gamma


class CrossMambaBlock(nn.Module):
    def __init__(self, dim, H, W, rank_ratio=0.5):
        super().__init__()
        self.norm0 = RMSNorm(dim)
        self.norm1 = RMSNorm(dim)
        self.block = Mamba(
            dim, expand=1, d_state=8, bimamba_type="v7", if_devide_out=True, use_norm=True, input_h=H, input_w=W
        )
        self.mlp = LowRankFeedForward(dim, rank_ratio=rank_ratio)

    def forward(self, x0, x1):
        residual = x0
        x0 = self.norm0(x0)
        x1 = self.norm1(x1)
        x = self.block(x0, extra_emb=x1)
        x = x + residual
        x = x + self.mlp(x)
        return x


class FusionMamba(nn.Module):
    def __init__(self, in_channels_pan, dim, H, W, depth=1, shared_weights=True):
        super().__init__()
        self.H, self.W, self.dim = H, W, dim

        def proj_layer(in_c):
            return nn.Sequential(
                nn.Conv2d(in_c, in_c, 3, padding=1, groups=in_c, bias=False),
                nn.Conv2d(in_c, dim, 1, bias=False),
                nn.SiLU(),
            )

        self.proj_pan = proj_layer(in_channels_pan)
        self.proj_ms = proj_layer(in_channels_pan)

        # Dynamic gated pooling
        self.gate = nn.Parameter(torch.tensor(0.5))
        self.avg_pool = nn.AdaptiveAvgPool2d((H, W))
        self.max_pool = nn.AdaptiveMaxPool2d((H, W))

        shared_block = (
            Mamba(dim, expand=1, d_state=8, bimamba_type="v6", if_devide_out=True, use_norm=True, input_h=H, input_w=W)
            if shared_weights
            else None
        )

        self.spa_layers = nn.ModuleList([SingleMambaBlock(dim, H, W, shared_block) for _ in range(depth)])
        self.spe_layers = nn.ModuleList([SingleMambaBlock(dim, H, W, shared_block) for _ in range(depth)])

        self.cross_spa = CrossMambaBlock(dim, H, W)
        self.cross_spe = CrossMambaBlock(dim, H, W)

        # Here we pass in channels dim*2 (because concatenating pan_out and ms_out)
        self.eca = ECABlock(dim * 2, k_size=3)
        self.out_conv = nn.Conv2d(dim * 2, dim, 1, bias=False)

    def forward(self, x):
        pan, ms = x
        b, _, h0, w0 = pan.shape

        pan = self.proj_pan(pan)
        ms = self.proj_ms(ms)

        pan_p = self.gate * self.avg_pool(pan) + (1 - self.gate) * self.max_pool(pan)
        ms_p = self.gate * self.avg_pool(ms) + (1 - self.gate) * self.max_pool(ms)

        pan_flat = rearrange(pan_p, "b c h w -> b (h w) c")
        ms_flat = rearrange(ms_p, "b c h w -> b (h w) c")

        for spa_layer, spe_layer in zip(self.spa_layers, self.spe_layers):
            pan_flat = spa_layer(pan_flat)
            ms_flat = spe_layer(ms_flat)

        spa_fused = self.cross_spa(pan_flat, ms_flat)
        spe_fused = self.cross_spe(ms_flat, pan_flat)

        pan_rec = rearrange(spa_fused, "b (h w) c -> b c h w", h=self.H, w=self.W)
        ms_rec = rearrange(spe_fused, "b (h w) c -> b c h w", h=self.H, w=self.W)

        pan_out = F.interpolate(pan_rec, size=(h0, w0), mode="bilinear", align_corners=False)
        ms_out = F.interpolate(ms_rec, size=(h0, w0), mode="bilinear", align_corners=False)

        fused = torch.cat([pan_out, ms_out], dim=1)  # (B, dim*2, H0, W0)
        fused = self.eca(fused)  # ECA expects ch=dim*2
        fused = self.out_conv(fused)

        return fused


# Hyprer-Mamba
# connection in series：x -> HyperACE -> Mamba -> residual
class SequentialHyperMamba(nn.Module):
    def __init__(
        self,
        c: int,
        H: int,
        W: int,
        num_hyperedges: int = 8,
        n: int = 1,
        dsc3k: bool = True,
        shortcut: bool = False,
        mamba_depth: int = 1,
        e1: float = 0.5,
        e2: float = 1.0,
        context: str = "both",
    ):
        super().__init__()
        self.c = c
        self.H = H
        self.W = W

        # 2D higher-order structure modeling
        self.hyper_ace = HyperACE(
            c1=c,
            c2=c,
            n=n,
            num_hyperedges=num_hyperedges,
            dsc3k=dsc3k,
            shortcut=shortcut,
            e1=e1,
            e2=e2,
            context=context,
        )

        # Sequence-based Mamba modeling (main computation)
        self.mamba_layers = nn.ModuleList([SingleMambaBlock(dim=c, H=H, W=W) for _ in range(mamba_depth)])

        # Lightweight output adjustment
        self.out_proj = Conv(c, c, k=1, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        """
        identity = x
        B, C, H, W = x.shape

        # HyperGraph: higher-order relations on 2D
        x_h = self.hyper_ace(x)  # [B, C, H, W]

        # Mamba: global modeling on flattened sequence
        seq = rearrange(x_h, "b c h w -> b (h w) c")  # [B, L, C], L=H*W
        for layer in self.mamba_layers:
            seq = layer(seq)
        x_m = rearrange(seq, "b (h w) c -> b c h w", h=H, w=W)

        # Residual fusion
        out = identity + x_h + x_m
        out = self.out_proj(out)
        return out


# structured: Mamba as backbone, HyperACE generates gate to modulate Mamba output
class StructuredHyperMamba(nn.Module):
    def __init__(
        self,
        c: int,
        H: int,
        W: int,
        num_hyperedges: int = 8,
        mamba_depth: int = 1,
        e1: float = 0.5,
        e2: float = 1.0,
        context: str = "both",
    ):
        super().__init__()
        self.c = c
        self.H = H
        self.W = W

        # Backbone: Mamba sequence modeling
        self.mamba_layers = nn.ModuleList([SingleMambaBlock(dim=c, H=H, W=W) for _ in range(mamba_depth)])

        # Structural information: HyperGraph
        self.hyper_branch = C3AH(self.c, self.c, e2, num_hyperedges, context)

        # Use HyperACE output to generate channel gate
        self.gate_conv = Conv(c, c, k=1, s=1)
        self.gate_act = nn.Sigmoid()

        # Output adjustment
        self.out_proj = Conv(c, c, k=1, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        """
        identity = x
        B, C, H, W = x.shape

        # Mamba backbone: directly model on x
        seq = rearrange(x, "b c h w -> b (h w) c")
        for layer in self.mamba_layers:
            seq = layer(seq)
        x_m = rearrange(seq, "b (h w) c -> b c h w", h=H, w=W)

        # HyperGraph generates gate
        x_h = self.hyper_branch(x)  # [B, C, H, W]
        gate = self.gate_act(self.gate_conv(x_h))  # [B, C, H, W]

        # Structural modulation + residual
        x_fused = gate * x_m
        out = identity + x_fused
        out = self.out_proj(out)
        return out


class CrossModalHyperMamba(nn.Module):
    def __init__(
        self,
        c: int,
        H: int,
        W: int,
        num_hyperedges: int = 8,
        mamba_depth: int = 1,
        e1: float = 0.5,
        e2: float = 1.0,
        context: str = "both",
    ):
        super().__init__()
        self.c = c
        self.H = H
        self.W = W

        # Cross-modal HyperGraph
        self.cmhyper = CHyperACE(
            c1=c,
            c2=c,
            n=1,
            num_hyperedges=num_hyperedges,
            dsc3k=True,
            shortcut=False,
            e1=e1,
            e2=e2,
            context=context,
        )

        # Mamba sequence modeling
        self.mamba_layers = nn.ModuleList([SingleMambaBlock(dim=c, H=H, W=W) for _ in range(mamba_depth)])

        # Output adjustment
        self.out_proj1 = Conv(c, c, k=1, s=1)
        self.out_proj2 = Conv(c, c, k=1, s=1)

        # Gate activation
        self.gate_act = nn.Sigmoid()

    def forward(self, x: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        x: list of [B, C, H, W], len=2
        """
        identity1 = x[0]
        identity2 = x[1]
        B, C, H, W = x[0].shape

        # Cross-modal HyperGraph
        x_h = self.cmhyper(x)  # [B, C, H, W]
        # Mamba sequence modeling
        seq = rearrange(x_h, "b c h w -> b (h w) c")
        for layer in self.mamba_layers:
            seq = layer(seq)
        x_m = rearrange(seq, "b (h w) c -> b c h w", h=H, w=W)

        # Residual fusion
        gate1 = self.gate_act(self.out_proj1(x_h))
        gate2 = self.gate_act(self.out_proj2(x_h))
        y1 = identity1 + gate1 * x_m
        y2 = identity2 + gate2 * x_m

        return y1, y2


class FourInputFusionBlock(nn.Module):
    def __init__(self, in_channels=1024, use_attn=True):
        super().__init__()
        if isinstance(in_channels, int):
            in_channels = [in_channels] * 3
        assert len(in_channels) == 3

        c = min(in_channels)  # The channel is compressed to 1/4
        self.proj = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(ic, c, 1, bias=False), nn.BatchNorm2d(c), nn.SiLU()) for ic in in_channels]
        )

        # Learnable fusion weights (lighter than concat)
        self.weights = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.post = GhostConv(c, c, k=3, p=1)

        # attentation
        self.attn = ECABlock(c) if use_attn else nn.Identity()

        self.out_bn = nn.BatchNorm2d(c)

    def forward(self, x):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]

        xs = [proj(inp) for proj, inp in zip(self.proj, (x1, x2, x3))]

        # Weighted Fusion (Adaptive Scaling)
        w = torch.softmax(self.weights, dim=0)
        feat = sum(w[i] * xs[i] for i in range(3))

        out = self.post(feat)
        out = self.attn(out)
        out = self.out_bn(out)
        return out
