from typing import Tuple, List

import math
import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv

from machine_learning.utils.logger import LOGGER


class DetectV8(nn.Module):
    """YOLOv8 Detect Head for object detection."""

    def __init__(
        self,
        nc: int = 80,  # number of classes
        ch: Tuple[int, ...] = (256, 512, 512),  # input channels for each scale
        reg_max: int = 16,  # DFL channels
    ):
        super().__init__()

        self.nc = nc  # number of classes
        self.reg_max = reg_max  # DFL parameters
        self.no = nc + reg_max * 4  # channels of each anchor (cls + reg)
        self.stride = None  # stride

        # Calculation of the number of channels in the middle layer
        dfl_hidden_ch = max(16, ch[0] // 4, self.reg_max * 4)  # The number of channels in the hidden layer of the DFL
        cls_hidden_ch = max(ch[0], self.nc)  # The number of channels in the hidden layer of the classification

        # DFL branch
        self.cv1 = nn.ModuleList(
            [
                nn.Sequential(
                    Conv(x, dfl_hidden_ch, 3),
                    Conv(dfl_hidden_ch, dfl_hidden_ch, 3),
                    nn.Conv2d(dfl_hidden_ch, 4 * self.reg_max, 1),
                )
                for x in ch
            ]
        )

        # cls branch
        self.cv2 = nn.ModuleList(
            [
                nn.Sequential(
                    Conv(x, cls_hidden_ch, 3),
                    Conv(cls_hidden_ch, cls_hidden_ch, 3),
                    nn.Conv2d(cls_hidden_ch, self.nc, 1),
                )
                for x in ch
            ]
        )

    def bias_init(self):
        """Initialize biases for classification branch."""
        LOGGER.info(f"Initializing bias parameters of {self.__class__.__name__}...")

        for a, b, s in zip(self.cv1, self.cv2, self.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: self.nc] = math.log(
                5 / self.nc / (640 / s) ** 2
            )  # cls (.01 objects, 80 classes, 640 img)

    def forward(self, x: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        """Forward pass through detection head."""
        return tuple(torch.cat([self.cv1[i](xi), self.cv2[i](xi)], 1) for i, xi in enumerate(x))
