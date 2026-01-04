from typing import Sequence, Literal

import torch
import torch.nn as nn

from machine_learning.networks import BaseNet
from machine_learning.modules.heads import DetectV8

from ultralytics.nn.modules import Conv, C2f, SPPF, Concat


class V8Net(BaseNet):
    def __init__(
        self,
        imgsz: Sequence[int],
        channel: int = 3,
        nc: int = 1,
        net_scale: Literal["n", "s", "l", "x"] = "n",
        *args,
        **kwargs,
    ):
        """Yolov8 object detection network.

        Args:
            imgsz (int): the size of input images.
            channels (int): number of input channels.
            nc (int): number of classes.
            net_scale (Literal["n", "s", "l", "x"]): scale of the network.
        """
        super().__init__(args=args, kwargs=kwargs)

        self.imgsz = imgsz
        self.nc = nc
        self.net_scale = net_scale
        self.in_channels = channel

        if self.net_scale == "n":
            # backbone
            self.backbone = nn.ModuleDict(
                {
                    "Conv_1": Conv(self.in_channels, 16, 3, 2),  # layer 0  (3, 640, 640) -> (16, 319, 319)
                    "Conv_2": Conv(16, 32, 3, 2),  # layer 1 (16, 319, 319) -> (32, 160, 160)
                    "C2f_1": C2f(32, 32, 1, True),  # layer 2 (32, 160, 160) -> (64, 160, 160)
                    "Conv_3": Conv(32, 64, 3, 2),  # layer 3 (64, 160, 160) -> (64, 80, 80)
                    "C2f_2": C2f(64, 64, 2, True),  # layer 4 b1-out (64, 80, 80)
                    "Conv_4": Conv(64, 128, 3, 2),  # layer 5
                    "C2f_3": C2f(128, 128, 2, True),  # layer 6 b2-out (128, 40, 40)
                    "Conv_5": Conv(128, 256, 3, 2),  # layer 7
                    "C2f_4": C2f(256, 256, 1, True),  # layer 8 b3-out (256, 20, 20)
                    "SPPF": SPPF(256, 256, k=5),  # layer 9 (256, 20, 20)
                }
            )
            # neck
            self.neck = nn.ModuleDict(
                {
                    # common ops
                    "Upsample": nn.Upsample(None, 2, "nearest"),
                    "Cat": Concat(1),
                    # trainable modules
                    "c2f_1": C2f(384, 128, 1, False),
                    "c2f_2": C2f(192, 64, 1, False),
                    "Conv_1": Conv(64, 64, 3, 2),
                    "c2f_3": C2f(192, 128, 1, False),
                    "Conv_2": Conv(128, 128, 3, 2),
                    "c2f_4": C2f(384, 256, 1, False),
                }
            )
            # head
            self.head = DetectV8(nc=self.nc, ch=(64, 128, 256))

        elif self.net_scale == "s":
            # backbone
            self.backbone = nn.ModuleDict(
                {
                    "Conv_1": Conv(self.in_channels, 32, 3, 2),  # layer 0  (3, 640, 640) -> (16, 319, 319)
                    "Conv_2": Conv(32, 64, 3, 2),  # layer 1 (16, 319, 319) -> (32, 160, 160)
                    "C2f_1": C2f(64, 64, 1, True),  # layer 2 (32, 160, 160) -> (64, 160, 160)
                    "Conv_3": Conv(64, 128, 3, 2),  # layer 3 (64, 160, 160) -> (128, 80, 80)
                    "C2f_2": C2f(128, 128, 2, True),  # layer 4 b1-out (128, 80, 80)
                    "Conv_4": Conv(128, 256, 3, 2),  # layer 5
                    "C2f_3": C2f(256, 256, 2, True),  # layer 6 b2-out (256, 40, 40)
                    "Conv_5": Conv(256, 512, 3, 2),
                    "C2f_4": C2f(512, 512, 1, True),  # layer 8 (512, 20, 20)
                    "SPPF": SPPF(512, 512, k=5),  # layer 9 b3-out (512, 20, 20)
                }
            )
            # neck
            self.neck = nn.ModuleDict(
                {
                    # common ops
                    "Upsample": nn.Upsample(None, 2, "nearest"),
                    "Cat": Concat(1),
                    # trainable modules
                    "c2f_1": C2f(768, 256, 1, False),
                    "c2f_2": C2f(384, 128, 1, False),
                    "Conv_1": Conv(128, 128, 3, 2),
                    "c2f_3": C2f(384, 256, 1, False),
                    "Conv_2": Conv(256, 256, 3, 2),
                    "c2f_4": C2f(768, 512, 1, False),
                }
            )
            # head
            self.head = DetectV8(nc=self.nc, ch=(128, 256, 512))

        elif self.net_scale == "m":
            # backbone
            self.backbone = nn.ModuleDict(
                {
                    "Conv_1": Conv(self.in_channels, 48, 3, 2),
                    "Conv_2": Conv(48, 96, 3, 2),
                    "C2f_1": C2f(96, 96, 2, True),
                    "Conv_3": Conv(96, 192, 3, 2),
                    "C2f_2": C2f(192, 192, 4, True),
                    "Conv_4": Conv(192, 384, 3, 2),
                    "C2f_3": C2f(384, 384, 4, True),
                    "Conv_5": Conv(384, 768, 3, 2),
                    "C2f_4": C2f(768, 768, 2, True),
                    "SPPF": SPPF(768, 768, k=5),
                }
            )
            # neck
            self.neck = nn.ModuleDict(
                {
                    # common ops
                    "Upsample": nn.Upsample(None, 2, "nearest"),
                    "Cat": Concat(1),
                    # trainable modules
                    "c2f_1": C2f(1152, 384, 2, False),
                    "c2f_2": C2f(576, 192, 2, False),
                    "Conv_1": Conv(192, 192, 3, 2),
                    "c2f_3": C2f(576, 384, 2, False),
                    "Conv_2": Conv(384, 384, 3, 2),
                    "c2f_4": C2f(1152, 768, 2, False),
                }
            )
            self.head = DetectV8(nc=self.nc, ch=(192, 384, 768))

        elif self.net_scale == "l":
            self.backbone = nn.ModuleDict(
                {
                    "Conv_1": Conv(self.in_channels, 64, 3, 2),
                    "Conv_2": Conv(64, 128, 3, 2),
                    "C2f_1": C2f(128, 128, 3, True),
                    "Conv_3": Conv(128, 256, 3, 2),
                    "C2f_2": C2f(256, 256, 6, True),
                    "Conv_4": Conv(256, 512, 3, 2),
                    "C2f_3": C2f(512, 512, 6, True),
                    "Conv_5": Conv(512, 512, 3, 2),
                    "C2f_4": C2f(512, 512, 3, True),
                    "SPPF": SPPF(512, 512, k=5),
                }
            )
            self.neck = nn.ModuleDict(
                {
                    # common ops
                    "Upsample": nn.Upsample(None, 2, "nearest"),
                    "Cat": Concat(1),
                    # trainable modules
                    "c2f_1": C2f(1024, 512, 3, False),
                    "c2f_2": C2f(768, 256, 3, False),
                    "Conv_1": Conv(256, 256, 3, 2),
                    "c2f_3": C2f(768, 512, 3, False),
                    "Conv_2": Conv(512, 512, 3, 2),
                    "c2f_4": C2f(1024, 512, 3, False),
                }
            )
            self.head = DetectV8(nc=self.nc, ch=(256, 512, 512))

        elif self.net_scale == "x":
            self.backbone = nn.ModuleDict(
                {
                    "Conv_1": Conv(self.in_channels, 80, 3, 2),
                    "Conv_2": Conv(80, 160, 3, 2),
                    "C2f_1": C2f(160, 160, 3, True),
                    "Conv_3": Conv(160, 320, 3, 2),
                    "C2f_2": C2f(320, 320, 6, True),
                    "Conv_4": Conv(320, 512, 3, 2),
                    "C2f_3": C2f(512, 512, 6, True),
                    "Conv_5": Conv(512, 512, 3, 2),
                    "C2f_4": C2f(512, 512, 3, True),
                    "SPPF": SPPF(512, 512, k=5),
                }
            )
            self.neck = nn.ModuleDict(
                {
                    "Upsample": nn.Upsample(None, 2, "nearest"),
                    "Cat": Concat(1),
                    "c2f_1": C2f(1024, 512, 3, False),
                    "c2f_2": C2f(832, 320, 3, False),
                    "Conv_1": Conv(320, 320, 3, 2),
                    "c2f_3": C2f(832, 512, 3, False),
                    "Conv_2": Conv(512, 512, 3, 2),
                    "c2f_4": C2f(1024, 512, 3, False),
                }
            )
            self.head = DetectV8(nc=self.nc, ch=(320, 512, 512))

        else:
            raise ValueError(f"Unsupported net_scale '{self.net_scale}' for YOLOv8 network.")

    @property
    def dummy_input(self) -> torch.Tensor:
        return torch.randn(1, self.in_channels, self.imgsz, self.imgsz, device=self.device)

    def forward(self, imgs: torch.Tensor) -> tuple[torch.Tensor]:
        # img backbone
        img_skips = []
        for key, layer in self.backbone.items():
            imgs = layer(imgs)
            if key in ["C2f_2", "C2f_3", "SPPF"]:
                img_skips.append(imgs)

        # ----- neck -----
        p3 = img_skips[0]  # P3/8
        p4 = img_skips[1]  # P4/16
        p5 = img_skips[2]  # P5/32

        # The first upsampling branch: starting from P5
        # Upsample P5, concatenate with P4
        x = self.neck.Upsample(p5)
        x = self.neck.Cat([x, p4])
        x = self.neck.c2f_1(x)

        # The second upsampling branch: continue upsampling, concatenate with P3
        y = self.neck.Upsample(x)
        y = self.neck.Cat([y, p3])
        y = self.neck.c2f_2(y)
        det1 = y  # P3/8-small

        # The first downsampling branch: from P3-small downsample, concatenate with middle layer
        z = self.neck.Conv_1(y)
        z = self.neck.Cat([z, x])
        z = self.neck.c2f_3(z)
        det2 = z  # P4/16-medium

        # The second downsampling branch: from P4-medium downsample, concatenate with P5
        w = self.neck.Conv_2(z)
        w = self.neck.Cat([w, p5])
        w = self.neck.c2f_4(w)
        det3 = w  # P5/32-large

        return self.head([det1, det2, det3])

    def _initialize_weights(self):
        """Initialize model weights to random values."""
        for m in self.modules():
            t = type(m)
            if t is nn.Conv2d:
                pass
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in {nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU}:
                m.inplace = True

        self._initialize_strides()
        self.head.bias_init()

    def _initialize_strides(self):
        self.stride = torch.tensor(
            [self.imgsz / x.shape[-2] for x in self.forward(self.dummy_input)], dtype=torch.int8, device=self.device
        )
        self.head.stride = self.stride
