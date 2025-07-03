import torch
import numpy as np
import torch.nn.functional as F
from machine_learning.types.aliases import FilePath


def read_img_to_normalize_tensor(img_path: FilePath, mean: list[float], std: list[float]) -> torch.Tensor:
    import cv2
    from torchvision.transforms import Compose, ToTensor, Normalize

    image = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)
    tfs = Compose([ToTensor(), Normalize(mean=mean, std=std)])

    return tfs(image)


def pad_to_square(img: torch.Tensor, pad_value: float | None = None) -> torch.Tensor:
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # 填充数值
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # 填充数 (左， 右， 上， 下， 前， 后)
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)

    img = F.pad(img, pad, "constant", value=pad_value)

    return img


def resize(image: torch.Tensor, size: int) -> torch.Tensor:
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image
