from typing import Tuple, List

import math
import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv


class DetectV8(nn.Module):
    """YOLOv8 Detect Head for object detection."""

    def __init__(
        self,
        nc: int = 80,  # number of classes
        ch: Tuple[int, ...] = (256, 512, 512),  # input channels for each scale
        reg_max: int = 16,  # DFL channels
        stride: Tuple[int, ...] = (8, 16, 32),  # feature map strides
    ):
        super().__init__()
        assert len(ch) == len(stride), "ch must be the same longth with stride."

        self.nc = nc  # number of classes
        self.reg_max = reg_max  # DFL parameters
        self.no = nc + reg_max * 4  # channels of each anchor (cls + reg)
        self.stride = stride  # stride

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

        # 分类分支
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
        print(f"[INFO] Initializing bias parameters of {self.__class__.__name__}...")

        for cv2 in self.cv2:
            m = cv2[-1]  # 最后一个卷积层
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                b = m.bias.data.view(-1)  # 展平便于索引
                # 目标检测常用初始化技巧
                b[:1] += math.log(5 / self.nc / (640 / self.stride[0]))  # obj
                b[1:] += math.log(0.6 / (self.nc - 0.99999))  # cls
                m.bias = torch.nn.Parameter(b.view_as(m.bias.data), requires_grad=True)

    def forward(self, x: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        """Forward pass through detection head."""
        return tuple(torch.cat([self.cv1[i](xi), self.cv2[i](xi)], 1) for i, xi in enumerate(x))
