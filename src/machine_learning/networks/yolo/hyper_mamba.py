from typing import Sequence, Literal

import torch
import torch.nn as nn

from machine_learning.networks import BaseNet
from machine_learning.modules.heads import DetectV8
from machine_learning.modules.blocks import CHyperACE, MMFullPAD_Tunnel, MFD, FusionMamba, FourInputFusionBlock

from ultralytics.nn.modules import (
    Conv,
    DSC3k2,
    DSConv,
    A2C2f,
    HyperACE,
    DownsampleConv,
    Concat,
    FullPAD_Tunnel,
    C2f,
    SPPF,
)


class HyperMambaNet(BaseNet):
    def __init__(
        self,
        img_shape: Sequence[int],
        ir_shape: Sequence[int],
        nc: int = 1,
        *args,
        **kwargs,
    ):
        """HyperMamba multimodal object detection network.

        Args:
            img_shape (Sequence[int]): the shape of input rgb image.
            ir_shape (Sequence[int]): the shape of input ir image.
            num_classes (int): number of classes.
        """
        super().__init__(img_shape, ir_shape, nc, *args, **kwargs)
