import math
import torch
import torch.nn as nn


class SinePositionalEncoding1D(nn.Module):
    """Standard sinusoidal positional encoding for 1D sequences."""

    def __init__(self, dim: int, max_len: int = 5000, fuse: bool = True) -> None:
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even for sinusoidal positional encoding."
        self.fuse = fuse

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, dim]
        self.register_buffer(
            "pe", pe
        )  # register as buffer to avoid being updated during training, e.g. runing mean/std in batchnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)] if self.fuse else self.pe[:, : x.size(1)]  # [B, L, dim]


class SinePositionalEncoding2D(nn.Module):
    """
    This is a standard version of the position encoding generalized to work on 2D images.
    Source: https://github.com/facebookresearch/Mask2Former.

    Note: PositionalEncoding2D only encodes the relative position of pixels, not execute position embedding on the
    input feature map.
    """

    def __init__(
        self,
        num_pos_feats: int = 64,
        temperature: int = 10000,
        normalize: bool = False,
        scale: float = None,
        fuse: bool = True,
    ) -> None:
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("Normalize should be True if scale is passed.")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.fuse = fuse

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        if padding_mask is None:
            padding_mask = torch.zeros(
                (x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool
            )  # [B, H, W]
        not_padding_mask = ~padding_mask
        y_embed = not_padding_mask.cumsum(1, dtype=torch.float32)  # [B, H, W]
        x_embed = not_padding_mask.cumsum(2, dtype=torch.float32)  # [B, H, W]
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale  # [B, H, W]
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale  # [B, H, W]

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = torch.pow(self.temperature, 2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t  # [B, H, W, NumPosFeats]
        pos_y = y_embed[:, :, :, None] / dim_t  # [B, H, W, NumPosFeats]
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # [B, 2*NumPosFeats, H, W]

        return pos + x if self.fuse else pos  # [B, 2*NumPosFeats, H, W]
