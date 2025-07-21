import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import DSC3k, DSBottleneck, C3AH


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
        return x + self.proj_out(h)  # 在数据传播过程中保留原始信息并增强全局依赖


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
        self.fuse = CFuseModule(c1)
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


class MMFullPAD_Tunnel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate1 = nn.Parameter(torch.tensor(0.0))
        self.gate2 = nn.Parameter(torch.tensor(0.0))
        self.gate3 = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        out = x[0] + self.gate1 * x[1] + self.gate2 * x[2] + self.gate3 * x[3]
        return out


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)
