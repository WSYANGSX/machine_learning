from typing import Sequence

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from mamba_ssm.modules.mamba_simple import Mamba

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import DSC3k, DSBottleneck


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
            ),  # H, W unchanged
            nn.GroupNorm(32, out_channels),
            nn.ReLU(),
            nn.Dropout1d(dropout),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1),  # H, W unchanged
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


# --------------------------------------------- Hypergraph attentation  ------------------------------------------------
class AdaHyperedgeGen(nn.Module):
    """
    Generates an adaptive hyperedge participation matrix from a set of vertex features.
    Source: https://github.com/iMoonLab/yolov13.

    This module implements the Adaptive Hyperedge Generation mechanism. It generates dynamic hyperedge prototypes
    based on the global context of the input nodes and calculates a continuous participation matrix (A)
    that defines the relationship between each vertex and each hyperedge.

    Attributes:
        node_dim (int): The feature dimension of each input node.
        num_hyperedges (int): The number of hyperedges to generate.
        num_heads (int, optional): The number of attention heads for multi-head similarity calculation. Defaults to 4.
        dropout (float, optional): The dropout rate applied to the logits. Defaults to 0.1.
        context (str, optional): The type of global context to use ('mean', 'max', or 'both'). Defaults to "both".

    Methods:
        forward: Takes a batch of vertex features and returns the participation matrix A.

    Examples:
        >>> import torch
        >>> model = AdaHyperedgeGen(node_dim=64, num_hyperedges=16, num_heads=4)
        >>> x = torch.randn(2, 100, 64)  # (Batch, Num_Nodes, Node_Dim)
        >>> A = model(x)
        >>> print(A.shape)
        torch.Size([2, 100, 16])
    """

    def __init__(self, node_dim, num_hyperedges, num_heads=4, dropout=0.1, context="both"):
        super().__init__()
        self.num_heads = num_heads
        self.num_hyperedges = num_hyperedges
        self.head_dim = node_dim // num_heads
        self.context = context

        self.prototype_base = nn.Parameter(torch.Tensor(num_hyperedges, node_dim))
        nn.init.xavier_uniform_(self.prototype_base)
        if context in ("mean", "max"):
            self.context_net = nn.Linear(node_dim, num_hyperedges * node_dim)
        elif context == "both":
            self.context_net = nn.Linear(2 * node_dim, num_hyperedges * node_dim)
        else:
            raise ValueError(f"Unsupported context '{context}'. Expected one of: 'mean', 'max', 'both'.")

        self.pre_head_proj = nn.Linear(node_dim, node_dim)

        self.dropout = nn.Dropout(dropout)
        self.scaling = math.sqrt(self.head_dim)

    def forward(self, X):
        B, N, D = X.shape
        if self.context == "mean":
            context_cat = X.mean(dim=1)
        elif self.context == "max":
            context_cat, _ = X.max(dim=1)
        else:
            avg_context = X.mean(dim=1)
            max_context, _ = X.max(dim=1)
            context_cat = torch.cat([avg_context, max_context], dim=-1)
        prototype_offsets = self.context_net(context_cat).view(B, self.num_hyperedges, D)
        prototypes = self.prototype_base.unsqueeze(0) + prototype_offsets

        X_proj = self.pre_head_proj(X)
        X_heads = X_proj.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        proto_heads = prototypes.view(B, self.num_hyperedges, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        X_heads_flat = X_heads.reshape(B * self.num_heads, N, self.head_dim)
        proto_heads_flat = proto_heads.reshape(B * self.num_heads, self.num_hyperedges, self.head_dim).transpose(1, 2)

        logits = torch.bmm(X_heads_flat, proto_heads_flat) / self.scaling
        logits = logits.view(B, self.num_heads, N, self.num_hyperedges).mean(dim=1)

        logits = self.dropout(logits)

        return F.softmax(logits, dim=1)


class AdaHGConv(nn.Module):
    """
    Performs the adaptive hypergraph convolution.
    Source: https://github.com/iMoonLab/yolov13.

    This module contains the two-stage message passing process of hypergraph convolution:
    1. Generates an adaptive participation matrix using AdaHyperedgeGen.
    2. Aggregates vertex features into hyperedge features (vertex-to-edge).
    3. Disseminates hyperedge features back to update vertex features (edge-to-vertex).
    A residual connection is added to the final output.

    Attributes:
        embed_dim (int): The feature dimension of the vertices.
        num_hyperedges (int, optional): The number of hyperedges for the internal generator. Defaults to 16.
        num_heads (int, optional): The number of attention heads for the internal generator. Defaults to 4.
        dropout (float, optional): The dropout rate for the internal generator. Defaults to 0.1.
        context (str, optional): The context type for the internal generator. Defaults to "both".

    Methods:
        forward: Performs the adaptive hypergraph convolution on a batch of vertex features.

    Examples:
        >>> import torch
        >>> model = AdaHGConv(embed_dim=128, num_hyperedges=16, num_heads=8)
        >>> x = torch.randn(2, 256, 128) # (Batch, Num_Nodes, Dim)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([2, 256, 128])
    """

    def __init__(self, embed_dim, num_hyperedges=16, num_heads=4, dropout=0.1, context="both"):
        super().__init__()
        self.edge_generator = AdaHyperedgeGen(embed_dim, num_hyperedges, num_heads, dropout, context)
        self.edge_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU())
        self.node_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU())

    def forward(self, X):
        A = self.edge_generator(X)

        He = torch.bmm(A.transpose(1, 2), X)
        He = self.edge_proj(He)

        X_new = torch.bmm(A, He)
        X_new = self.node_proj(X_new)

        return X_new + X


class AdaHGComputation(nn.Module):
    """
    A wrapper module for applying adaptive hypergraph convolution to 4D feature maps.
    Source: https://github.com/iMoonLab/yolov13.

    This class makes the hypergraph convolution compatible with standard CNN architectures. It flattens a
    4D input tensor (B, C, H, W) into a sequence of vertices (tokens), applies the AdaHGConv layer to
    model high-order correlations, and then reshapes the output back into a 4D tensor.

    Attributes:
        embed_dim (int): The feature dimension of the vertices (equivalent to input channels C).
        num_hyperedges (int, optional): The number of hyperedges for the underlying AdaHGConv. Defaults to 16.
        num_heads (int, optional): The number of attention heads for the underlying AdaHGConv. Defaults to 8.
        dropout (float, optional): The dropout rate for the underlying AdaHGConv. Defaults to 0.1.
        context (str, optional): The context type for the underlying AdaHGConv. Defaults to "both".

    Methods:
        forward: Processes a 4D feature map through the adaptive hypergraph computation layer.

    Examples:
        >>> import torch
        >>> model = AdaHGComputation(embed_dim=64, num_hyperedges=8, num_heads=4)
        >>> x = torch.randn(2, 64, 32, 32) # (B, C, H, W)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([2, 64, 32, 32])
    """

    def __init__(self, embed_dim, num_hyperedges=16, num_heads=8, dropout=0.1, context="both"):
        super().__init__()
        self.embed_dim = embed_dim
        self.hgnn = AdaHGConv(
            embed_dim=embed_dim, num_hyperedges=num_hyperedges, num_heads=num_heads, dropout=dropout, context=context
        )

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.hgnn(tokens)
        x_out = tokens.transpose(1, 2).view(B, C, H, W)
        return x_out


class C3AH(nn.Module):
    """
    A CSP-style block integrating Adaptive Hypergraph Computation (C3AH).
    Source: https://github.com/iMoonLab/yolov13.

    The input feature map is split into two paths.
    One path is processed by the AdaHGComputation module to model high-order correlations, while the other
    serves as a shortcut. The outputs are then concatenated to fuse features.

    Attributes:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        e (float, optional): Expansion ratio for the hidden channels. Defaults to 1.0.
        num_hyperedges (int, optional): The number of hyperedges for the internal AdaHGComputation. Defaults to 8.
        context (str, optional): The context type for the internal AdaHGComputation. Defaults to "both".

    Methods:
        forward: Performs a forward pass through the C3AH module.

    Examples:
        >>> import torch
        >>> model = C3AH(c1=64, c2=128, num_hyperedges=8)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([2, 128, 32, 32])
    """

    def __init__(self, c1, c2, e=1.0, num_hyperedges=8, context="both"):
        super().__init__()
        c_ = int(c2 * e)
        assert c_ % 16 == 0, "Dimension of AdaHGComputation should be a multiple of 16."
        num_heads = c_ // 16
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = AdaHGComputation(
            embed_dim=c_, num_hyperedges=num_hyperedges, num_heads=num_heads, dropout=0.1, context=context
        )
        self.cv3 = Conv(2 * c_, c2, 1)

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class FuseModule(nn.Module):
    """
    A module to fuse multi-scale features for the HyperACE block.
    Source: https://github.com/iMoonLab/yolov13.

    This module takes a list of three feature maps from different scales, aligns them to a common
    spatial resolution by downsampling the first and upsampling the third, and then concatenates
    and fuses them with a convolution layer.

    Attributes:
        c_in (int): The number of channels of the input feature maps.
        channel_adjust (bool): Whether to adjust the channel count of the concatenated features.

    Methods:
        forward: Fuses a list of three multi-scale feature maps.

    Examples:
        >>> import torch
        >>> model = FuseModule(c_in=64, channel_adjust=False)
        >>> # Input is a list of features from different backbone stages
        >>> x_list = [torch.randn(2, 64, 64, 64), torch.randn(2, 64, 32, 32), torch.randn(2, 64, 16, 16)]
        >>> output = model(x_list)
        >>> print(output.shape)
        torch.Size([2, 64, 32, 32])
    """

    def __init__(self, c_in, channel_adjust):
        super(FuseModule, self).__init__()
        self.downsample = nn.AvgPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        if channel_adjust:
            self.conv_out = Conv(4 * c_in, c_in, 1)
        else:
            self.conv_out = Conv(3 * c_in, c_in, 1)

    def forward(self, x):
        x1_ds = self.downsample(x[0])
        x3_up = self.upsample(x[2])
        x_cat = torch.cat([x1_ds, x[1], x3_up], dim=1)
        out = self.conv_out(x_cat)
        return out


class HyperACE(nn.Module):
    """
    Hypergraph-based Adaptive Correlation Enhancement (HyperACE).
    Source: https://github.com/iMoonLab/yolov13.

    This is the core module of YOLOv13, designed to model both global high-order correlations and
    local low-order correlations. It first fuses multi-scale features, then processes them through parallel
    branches: two C3AH branches for high-order modeling and a lightweight DSConv-based branch for
    low-order feature extraction.

    Attributes:
        c1 (int): Number of input channels for the fuse module.
        c2 (int): Number of output channels for the entire block.
        n (int, optional): Number of blocks in the low-order branch. Defaults to 1.
        num_hyperedges (int, optional): Number of hyperedges for the C3AH branches. Defaults to 8.
        dsc3k (bool, optional): If True, use DSC3k in the low-order branch; otherwise, use DSBottleneck. Defaults to True.
        shortcut (bool, optional): Whether to use shortcuts in the low-order branch. Defaults to False.
        e1 (float, optional): Expansion ratio for the main hidden channels. Defaults to 0.5.
        e2 (float, optional): Expansion ratio within the C3AH branches. Defaults to 1.
        context (str, optional): Context type for C3AH branches. Defaults to "both".
        channel_adjust (bool, optional): Passed to FuseModule for channel configuration. Defaults to True.

    Methods:
        forward: Performs a forward pass through the HyperACE module.

    Examples:
        >>> import torch
        >>> model = HyperACE(c1=64, c2=256, n=1, num_hyperedges=8)
        >>> x_list = [torch.randn(2, 64, 64, 64), torch.randn(2, 64, 32, 32), torch.randn(2, 64, 16, 16)]
        >>> output = model(x_list)
        >>> print(output.shape)
        torch.Size([2, 256, 32, 32])
    """

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
        channel_adjust=True,
    ):
        super().__init__()
        self.c = int(c2 * e1)
        self.cv1 = Conv(c1, 3 * self.c, 1, 1)
        self.cv2 = Conv((4 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            DSC3k(self.c, self.c, 2, shortcut, k1=3, k2=7) if dsc3k else DSBottleneck(self.c, self.c, shortcut=shortcut)
            for _ in range(n)
        )
        self.fuse = FuseSEModule(c1, channel_adjust)
        self.branch1 = C3AH(self.c, self.c, e2, num_hyperedges, context)
        self.branch2 = C3AH(self.c, self.c, e2, num_hyperedges, context)

    def forward(self, X):
        x = self.fuse(X)
        y = list(self.cv1(x).chunk(3, 1))
        out1 = self.branch1(y[1])
        out2 = self.branch2(y[1])
        y.extend(m(y[-1]) for m in self.m)
        y[1] = out1
        y.append(out2)
        return self.cv2(torch.cat(y, 1))


class DownsampleConv(nn.Module):
    """
    A simple downsampling block with optional channel adjustment.
    Source: https://github.com/iMoonLab/yolov13.

    This module uses average pooling to reduce the spatial dimensions (H, W) by a factor of 2. It can
    optionally include a 1x1 convolution to adjust the number of channels, typically doubling them.

    Attributes:
        in_channels (int): The number of input channels.
        channel_adjust (bool, optional): If True, a 1x1 convolution doubles the channel dimension. Defaults to True.

    Methods:
        forward: Performs the downsampling and optional channel adjustment.

    Examples:
        >>> import torch
        >>> model = DownsampleConv(in_channels=64, channel_adjust=True)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([2, 128, 16, 16])
    """

    def __init__(self, in_channels, channel_adjust=True):
        super().__init__()
        self.downsample = nn.AvgPool2d(kernel_size=2)
        if channel_adjust:
            self.channel_adjust = Conv(in_channels, in_channels * 2, 1)
        else:
            self.channel_adjust = nn.Identity()

    def forward(self, x):
        return self.channel_adjust(self.downsample(x))


class FullPAD_Tunnel(nn.Module):
    """
    A gated fusion module for the Full-Pipeline Aggregation-and-Distribution (FullPAD) paradigm.
    Source: https://github.com/iMoonLab/yolov13.

    This module implements a gated residual connection used to fuse features. It takes two inputs: the original
    feature map and a correlation-enhanced feature map. It then computes `output = original + gate * enhanced`,
    where `gate` is a learnable scalar parameter that adaptively balances the contribution of the enhanced features.

    Methods:
        forward: Performs the gated fusion of two input feature maps.

    Examples:
        >>> import torch
        >>> model = FullPAD_Tunnel()
        >>> original_feature = torch.randn(2, 64, 32, 32)
        >>> enhanced_feature = torch.randn(2, 64, 32, 32)
        >>> output = model([original_feature, enhanced_feature])
        >>> print(output.shape)
        torch.Size([2, 64, 32, 32])
    """

    def __init__(self):
        super().__init__()
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        out = x[0] + self.gate * x[1]
        return out


class FuseSEModule(nn.Module):
    """
    A module to fuse multi-scale features for the HyperACE block.

    This module takes a list of three feature maps from different scales, aligns them to a common
    spatial resolution by downsampling the first and upsampling the third, and then concatenates
    and fuses them with a convolution layer.

    Attributes:
        c_in (int): The number of channels of the input feature maps.
        channel_adjust (bool): Whether to adjust the channel count of the concatenated features.

    Methods:
        forward: Fuses a list of three multi-scale feature maps.

    Examples:
        >>> import torch
        >>> model = FuseModule(c_in=64, channel_adjust=False)
        >>> # Input is a list of features from different backbone stages
        >>> x_list = [torch.randn(2, 64, 64, 64), torch.randn(2, 64, 32, 32), torch.randn(2, 64, 16, 16)]
        >>> output = model(x_list)
        >>> print(output.shape)
        torch.Size([2, 64, 32, 32])
    """

    def __init__(self, c_in, channel_adjust, reduction: int = 16):
        super().__init__()
        self.downsample = nn.AvgPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        if channel_adjust:
            self.conv_out = Conv(4 * c_in, c_in, 1)
        else:
            self.conv_out = Conv(3 * c_in, c_in, 1)

        # simple SE
        hidden = max(c_in // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_in, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c_in, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x1_ds = self.downsample(x[0])
        x3_up = self.upsample(x[2])
        x_cat = torch.cat([x1_ds, x[1], x3_up], dim=1)
        out = self.conv_out(x_cat)

        # gate attentation
        w = self.se(out)
        out = out * w

        return out


class ModalFuseSE(nn.Module):
    def __init__(self, c, reduction=16):
        super().__init__()
        # Concat it to 2c, and then use 1×1 Conv to press it back to the c dimension
        self.conv1x1 = Conv(2 * c, c, k=1, s=1)
        # Simple SE channel attention
        rc = max(c // reduction, 4)  # Prevent it from being too small
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, rc, 1, 1, 0),
            nn.SiLU(inplace=True),
            nn.Conv2d(rc, c, 1, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, x0, x1):
        x = torch.cat([x0, x1], dim=1)  # [B,2C,H,W]
        x = self.conv1x1(x)  # [B,C,H,W]
        w = self.attn(x)  # [B,C,1,1]
        return x * w  # The fusion features after channel recalibration


class ModalFuseGate(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        # add gate weight
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

        self.conv_out = Conv(2 * c_in, c_in, 1)

    def forward(self, x):
        x0, x1 = x

        # gate
        x0_w = self.alpha * x0
        x1_w = self.beta * x1

        x_cat = torch.cat([x0_w, x1_w], dim=1)
        out = self.conv_out(x_cat)
        return out


class M2CA(nn.Module):
    """Multi modal channels attentation."""

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
        self.fuse = ModalFuseGate(c1)
        self.branch1 = C3AH(self.c, self.c, e2, num_hyperedges, context)
        self.branch2 = C3AH(self.c, self.c, e2, num_hyperedges, context)

    def forward(self, X):
        x = self.fuse(X)
        y = list(self.cv1(x).chunk(3, 1))
        out1 = self.branch1(y[1])
        out2 = self.branch2(y[1])
        y.extend(m(y[-1]) for m in self.m)
        y[1] = out1
        y.append(out2)

        return self.cv2(torch.cat(y, 1))


class M2CAHyperACE(nn.Module):
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
        self.cmca = M2CA(c1, c1)
        self.fuse = ModalFuseGate(c1)
        self.branch1 = C3AH(self.c, self.c, e2, num_hyperedges, context)
        self.branch2 = C3AH(self.c, self.c, e2, num_hyperedges, context)

    def forward(self, X):
        X_hat = self.cmca(X)
        x = self.fuse(X_hat)
        y = list(self.cv1(x).chunk(3, 1))
        out1 = self.branch1(y[1])
        out2 = self.branch2(y[1])
        y.extend(m(y[-1]) for m in self.m)
        y[1] = out1
        y.append(out2)

        return self.cv2(torch.cat(y, 1))


class HierarchicalHyperedgeGen(nn.Module):
    

class IntraHyperEnhance(nn.Module):
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


class IntreHyperFusion(nn.Module):
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


class MMFullPAD_Tunnel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate1 = nn.Parameter(torch.tensor(0.0))
        self.gate2 = nn.Parameter(torch.tensor(0.0))
        self.gate3 = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        out = x[0] + self.gate1 * x[1] + self.gate2 * x[2] + self.gate3 * x[3]
        return out


# ----------------------------------------------- Mamba attentation  ---------------------------------------------------
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


class GhostConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=0, ratio=2, act=True):
        super().__init__()
        init_channels = math.ceil(out_ch / ratio)
        new_channels = init_channels * (ratio - 1)
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_ch, init_channels, k, s, p, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.SiLU() if act else nn.Identity(),
        )
        self.cheap_op = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, 3, 1, 1, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.SiLU() if act else nn.Identity(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_op(x1)
        return torch.cat([x1, x2], dim=1)[:, : self.primary_conv[0].out_channels * 2, :, :]


# Single Mamba module
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


# Cross Mamba module
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


# Fusion module
class FusionMamba(nn.Module):
    def __init__(self, in_channels_pan, dim, H, W, depth=1, shared_weights=True):
        super().__init__()
        self.H, self.W, self.dim = H, W, dim

        # Mapping layer: Depthwise + Pointwise (retains lightweight)
        def proj_layer(in_c):
            return nn.Sequential(
                nn.Conv2d(in_c, in_c, 3, padding=1, groups=in_c, bias=False),
                nn.Conv2d(in_c, dim, 1, bias=False),
                nn.SiLU(),
            )

        self.proj_pan = proj_layer(in_channels_pan)
        self.proj_ms = proj_layer(in_channels_pan)

        # Dynamic gating pooling
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

        # The channel dim*2 is passed in (because pan out and ms out are concatenated)
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


class FourInputFusionBlock(nn.Module):
    def __init__(self, in_channels=1024, use_attn=True):
        super().__init__()
        if isinstance(in_channels, int):
            in_channels = [in_channels] * 3
        assert len(in_channels) == 3

        c = min(in_channels)  # Channel compression to 1/4
        self.proj = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(ic, c, 1, bias=False), nn.BatchNorm2d(c), nn.SiLU()) for ic in in_channels]
        )

        # Learnable fusion weights (lighter than concat)
        self.weights = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.post = GhostConv(c, c, k=3, p=1)

        # Attention
        self.attn = ECABlock(c) if use_attn else nn.Identity()

        # Output standardization
        self.out_bn = nn.BatchNorm2d(c)

    def forward(self, x):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]

        xs = [proj(inp) for proj, inp in zip(self.proj, (x1, x2, x3))]

        # Weighted fusion (Adaptive scaling)
        w = torch.softmax(self.weights, dim=0)
        feat = sum(w[i] * xs[i] for i in range(3))

        out = self.post(feat)
        out = self.attn(out)
        out = self.out_bn(out)
        return out


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
