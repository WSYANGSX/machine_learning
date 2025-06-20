import torch
import numpy as np
import torch.nn.functional as F


def pad_to_square(img: torch.Tensor, pad_value: float | None = None) -> torch.Tensor:
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # 填充数值
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # 填充数 (左， 右， 上， 下， 前， 后)
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)

    img = F.pad(img, pad, "constant", value=pad_value)

    return img


def resize(image: torch.Tensor, size) -> torch.Tensor:
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image
