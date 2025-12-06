from typing import Sequence, Literal

import torch
import torch.nn as nn

from machine_learning.networks import BaseNet
from machine_learning.modules.heads import DetectV8
from machine_learning.modules.blocks import CHyperACE, MMFullPAD_Tunnel, MFD, FusionMamba, FourInputFusionBlock


class SuperYoloNet(BaseNet):
    def __init__(
        self,
        img_shape: Sequence[int],
        ir_shape: Sequence[int],
        nc: int = 1,
        *args,
        **kwargs,
    ):
        """SuperYolo multimodal object detection network.

        Args:
            img_shape (Sequence[int]): the shape of input rgb image.
            ir_shape (Sequence[int]): the shape of input ir image.
            num_classes (int): number of classes.
        """
        super().__init__(img_shape, ir_shape, nc, *args, **kwargs)
