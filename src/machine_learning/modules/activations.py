"""Custom activation functions for neural networks."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit.
    Splits the input tensor in half along the specified dimension.
    """

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=self.dim)
        return x * F.silu(gate)


class GEGLU(nn.Module):
    """
    GELU-Gated Linear Unit.
    Splits the input tensor in half along the specified dimension.
    """

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=self.dim)
        return x * F.gelu(gate)
