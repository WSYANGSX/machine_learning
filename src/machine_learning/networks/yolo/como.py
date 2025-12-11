from typing import Literal, Any

import torch
import torch.nn as nn

try:
    from thop import profile  # pip install thop
except ImportError:
    profile = None

from machine_learning.networks import BaseNet
from machine_learning.modules.heads import DetectV8
from machine_learning.modules.blocks import FusionMamba, FourInputFusionBlock
from ultralytics.nn.modules import Conv, Concat, C2f, SPPF


class COMONet(BaseNet):
    def __init__(
        self,
        imgsz: int,
        channels: int = 3,
        nc: int = 1,
        net_scale: Literal["n", "s", "m", "l", "x"] = "n",
        *args: tuple[Any, ...],
        **kwargs: dict[str, Any],
    ):
        """
        COMONet: multimodal object detection network (RGB + IR), https://github.com/luluyuu/COMO.

        Args:
            imgsz (int): Input the image size.
            channels (int): The number of input channels for each mode.
            nc (int): number of classes.
            net_scale: The scale of the net.
        """
        super().__init__(args=args, kwargs=kwargs)

        self.imgsz = imgsz
        self.nc = nc
        self.channels = channels
        self.net_scale = net_scale

        if self.net_scale == "n":
            # img backbone
            self.img_backbone = nn.ModuleDict(
                {
                    "Conv_0": Conv(self.channels, 16, 6, 2, 2),  # 0-P1/2 (3, 640, 640) -> (16, 320, 320)
                    "Conv_1": Conv(16, 32, 3, 2),  # 1-P2/4 (16, 320, 320) -> (32, 160, 160)
                    "C2f_2": C2f(32, 32, n=1, shortcut=True),  # 2 (P2) (32, 160, 160)
                    "Conv_3": Conv(32, 64, 3, 2),  # 3-P3/8 (32, 160, 160) -> (64, 80, 80)
                    "C2f_4": C2f(64, 64, n=2, shortcut=True),  # 4 (P3) (64, 80, 80)
                    "Conv_5": Conv(64, 128, 3, 2),  # 5-P4/16 (64, 80, 80) -> (128, 40, 40)
                    "C2f_6": C2f(128, 128, n=3, shortcut=True),  # 6 (P4) (128, 40, 40)
                    "Conv_7": Conv(128, 256, 3, 2),  # 7-P5/32 (128, 40, 40) -> (256, 20, 20)
                    "C2f_8": C2f(256, 256, n=1, shortcut=True),  # 8 (P5) (256, 20, 20)
                    "SPPF_9": SPPF(256, 256, k=5),  # 9 (P5) (256, 20, 20)
                }
            )
            # ir backbone
            self.ir_backbone = nn.ModuleDict(
                {
                    "Conv_0": Conv(self.channels, 16, 6, 2, 2),  # 10-P1/2 (3, 640, 640) -> (16, 320, 320)
                    "Conv_1": Conv(16, 32, 3, 2),  # 11-P2/4 (16, 320, 320) -> (32, 160, 160)
                    "C2f_2": C2f(32, 32, n=1, shortcut=True),  # 12 (P2) (32, 160, 160)
                    "Conv_3": Conv(32, 64, 3, 2),  # 13-P3/8 (32, 160, 160) -> (64, 80, 80)
                    "C2f_4": C2f(64, 64, n=2, shortcut=True),  # 14 (P3) (64, 80, 80)
                    "Conv_5": Conv(64, 128, 3, 2),  # 15-P4/16 (64, 80, 80) -> (128, 40, 40)
                    "C2f_6": C2f(128, 128, n=3, shortcut=True),  # 16 (P4) (128, 40, 40)
                    "Conv_7": Conv(128, 256, 3, 2),  # 17-P5/32 (128, 40, 40) -> (256, 20, 20)
                    "C2f_8": C2f(256, 256, n=1, shortcut=True),  # 18 (P5) (256, 20, 20)
                    "SPPF_9": SPPF(256, 256, k=5),  # 19 (P5) (256, 20, 20)
                }
            )
            # neck
            self.neck = nn.ModuleDict(
                {
                    # multi modal fusion
                    "FusionMamba": FusionMamba(256, 256, 8, 8),  # 20: from [P5_img, P5_ir]
                    "FourInputFusionBlock": FourInputFusionBlock(256),  # 21
                    # common ops
                    "Concat": Concat(1),
                    "Upsample": nn.Upsample(scale_factor=2, mode="nearest"),
                    # FPN
                    "Conv_fuse_p5": Conv(256, 128, 1, 1),
                    "C2f_p4": C2f(128 * 3, 128, n=1, shortcut=False),
                    "Conv_p4_to_p3": Conv(128, 64, 1, 1),
                    "C2f_p3": C2f(64 * 3, 64, n=1, shortcut=False),
                    # PAN`
                    "Conv_p3_down": Conv(64, 64, 3, 2),
                    "C2f_pan_p4": C2f(64 + 128, 128, n=1, shortcut=False),
                    "Conv_p4_down": Conv(128, 128, 3, 2),
                    "C2f_pan_p5": C2f(128 + 256, 256, n=1, shortcut=False),
                }
            )
            # head
            self.head = DetectV8(nc=self.nc, ch=(64, 128, 256))

        elif self.net_scale == "s":
            # img backbone
            self.img_backbone = nn.ModuleDict(
                {
                    "Conv_0": Conv(self.channels, 32, 6, 2, 2),  # 0-P1/2 (3, 640, 640) -> (32, 320, 320)
                    "Conv_1": Conv(32, 64, 3, 2),  # 1-P2/4 (32, 320, 320) -> (64, 160, 160)
                    "C2f_2": C2f(64, 64, n=1, shortcut=True),  # 2 (P2) (64, 160, 160)
                    "Conv_3": Conv(64, 128, 3, 2),  # 3-P3/8 (64, 160, 160) -> (128, 80, 80)
                    "C2f_4": C2f(128, 128, n=2, shortcut=True),  # 4 (P3) (128, 80, 80)
                    "Conv_5": Conv(128, 256, 3, 2),  # 5-P4/16 (128, 80, 80) -> (256, 40, 40)
                    "C2f_6": C2f(256, 256, n=3, shortcut=True),  # 6 (P4) (256, 40, 40)
                    "Conv_7": Conv(256, 512, 3, 2),  # 7-P5/32 (256, 40, 40) -> (512, 20, 20)
                    "C2f_8": C2f(512, 512, n=1, shortcut=True),  # 8 (P5) (512, 20, 20)
                    "SPPF_9": SPPF(512, 512, k=5),  # 9 (P5) (512, 20, 20)
                }
            )
            # ir backbone
            self.ir_backbone = nn.ModuleDict(
                {
                    "Conv_0": Conv(self.channels, 32, 6, 2, 2),  # 0-P1/2 (3, 640, 640) -> (32, 320, 320)
                    "Conv_1": Conv(32, 64, 3, 2),  # 1-P2/4 (32, 320, 320) -> (64, 160, 160)
                    "C2f_2": C2f(64, 64, n=1, shortcut=True),  # 2 (P2) (64, 160, 160)
                    "Conv_3": Conv(64, 128, 3, 2),  # 3-P3/8 (64, 160, 160) -> (128, 80, 80)
                    "C2f_4": C2f(128, 128, n=2, shortcut=True),  # 4 (P3) (128, 80, 80)
                    "Conv_5": Conv(128, 256, 3, 2),  # 5-P4/16 (128, 80, 80) -> (256, 40, 40)
                    "C2f_6": C2f(256, 256, n=3, shortcut=True),  # 6 (P4) (256, 40, 40)
                    "Conv_7": Conv(256, 512, 3, 2),  # 7-P5/32 (256, 40, 40) -> (512, 20, 20)
                    "C2f_8": C2f(512, 512, n=1, shortcut=True),  # 8 (P5) (512, 20, 20)
                    "SPPF_9": SPPF(512, 512, k=5),  # 9 (P5) (512, 20, 20)
                }
            )
            # neck
            self.neck = nn.ModuleDict(
                {
                    # multi modal fusion
                    "FusionMamba": FusionMamba(512, 512, 8, 8),  # 20: from [P5_img, P5_ir]
                    "FourInputFusionBlock": FourInputFusionBlock(512),  # 21
                    # common ops
                    "Concat": Concat(1),
                    "Upsample": nn.Upsample(scale_factor=2, mode="nearest"),
                    # FPN
                    "Conv_fuse_p5": Conv(512, 256, 1, 1),
                    "C2f_p4": C2f(768, 256, n=1, shortcut=False),
                    "Conv_p4_to_p3": Conv(256, 128, 1, 1),
                    "C2f_p3": C2f(384, 128, n=1, shortcut=False),
                    # PAN`
                    "Conv_p3_down": Conv(128, 128, 3, 2),
                    "C2f_pan_p4": C2f(256, 256, n=1, shortcut=False),
                    "Conv_p4_down": Conv(256, 256, 3, 2),
                    "C2f_pan_p5": C2f(768, 512, n=1, shortcut=False),
                }
            )
            # head
            self.head = DetectV8(nc=self.nc, ch=(128, 256, 512))

        elif self.net_scale == "m":
            # img backbone
            self.img_backbone = nn.ModuleDict(
                {
                    "Conv_0": Conv(self.channels, 48, 6, 2, 2),  # 0-P1/2 (3, 640, 640) -> (48, 320, 320)
                    "Conv_1": Conv(48, 96, 3, 2),  # 1-P2/4 (48, 320, 320) -> (96, 160, 160)
                    "C2f_2": C2f(96, 96, n=2, shortcut=True),  # 2 (P2) (96, 160, 160)
                    "Conv_3": Conv(96, 192, 3, 2),  # 3-P3/8 (96, 160, 160) -> (192, 80, 80)
                    "C2f_4": C2f(192, 192, n=4, shortcut=True),  # 4 (P3) (192, 80, 80)
                    "Conv_5": Conv(192, 384, 3, 2),  # 5-P4/16 (192, 80, 80) -> (384, 40, 40)
                    "C2f_6": C2f(384, 384, n=6, shortcut=True),  # 6 (P4) (384, 40, 40)
                    "Conv_7": Conv(384, 768, 3, 2),  # 7-P5/32 (384, 40, 40) -> (768, 20, 20)
                    "C2f_8": C2f(768, 768, n=2, shortcut=True),  # 8 (P5) (768, 20, 20)
                    "SPPF_9": SPPF(768, 768, k=5),  # 9 (P5) (768, 20, 20)
                }
            )
            # ir backbone
            self.ir_backbone = nn.ModuleDict(
                {
                    "Conv_0": Conv(self.channels, 48, 6, 2, 2),  # 0-P1/2 (3, 640, 640) -> (48, 320, 320)
                    "Conv_1": Conv(48, 96, 3, 2),  # 1-P2/4 (48, 320, 320) -> (96, 160, 160)
                    "C2f_2": C2f(96, 96, n=2, shortcut=True),  # 2 (P2) (96, 160, 160)
                    "Conv_3": Conv(96, 192, 3, 2),  # 3-P3/8 (96, 160, 160) -> (192, 80, 80)
                    "C2f_4": C2f(192, 192, n=4, shortcut=True),  # 4 (P3) (192, 80, 80)
                    "Conv_5": Conv(192, 384, 3, 2),  # 5-P4/16 (192, 80, 80) -> (384, 40, 40)
                    "C2f_6": C2f(384, 384, n=6, shortcut=True),  # 6 (P4) (384, 40, 40)
                    "Conv_7": Conv(384, 768, 3, 2),  # 7-P5/32 (384, 40, 40) -> (768, 20, 20)
                    "C2f_8": C2f(768, 768, n=2, shortcut=True),  # 8 (P5) (768, 20, 20)
                    "SPPF_9": SPPF(768, 768, k=5),  # 9 (P5) (768, 20, 20)
                }
            )
            # neck
            self.neck = nn.ModuleDict(
                {
                    # multi modal fusion
                    "FusionMamba": FusionMamba(768, 768, 8, 8),  # 20: from [P5_img, P5_ir]
                    "FourInputFusionBlock": FourInputFusionBlock(768),  # 21
                    # common ops
                    "Concat": Concat(1),
                    "Upsample": nn.Upsample(scale_factor=2, mode="nearest"),
                    # FPN
                    "Conv_fuse_p5": Conv(768, 384, 1, 1),
                    "C2f_p4": C2f(384 * 3, 384, n=2, shortcut=False),
                    "Conv_p4_to_p3": Conv(384, 192, 1, 1),
                    "C2f_p3": C2f(192 * 3, 192, n=2, shortcut=False),
                    # PAN`
                    "Conv_p3_down": Conv(192, 192, 3, 2),
                    "C2f_pan_p4": C2f(192 * 3, 384, n=2, shortcut=False),
                    "Conv_p4_down": Conv(384, 384, 3, 2),
                    "C2f_pan_p5": C2f(384 + 768, 768, n=2, shortcut=False),
                }
            )
            # head
            self.head = DetectV8(nc=self.nc, ch=(192, 384, 768))

        elif self.net_scale == "l":
            # img backbone
            self.img_backbone = nn.ModuleDict(
                {
                    "Conv_0": Conv(self.channels, 64, 6, 2, 2),  # 0-P1/2 (3, 640, 640) -> (16, 320, 320)
                    "Conv_1": Conv(64, 128, 3, 2),  # 1-P2/4 (16, 320, 320) -> (32, 160, 160)
                    "C2f_2": C2f(128, 128, n=1, shortcut=True),  # 2 (P2) (32, 160, 160)
                    "Conv_3": Conv(128, 256, 3, 2),  # 3-P3/8 (32, 160, 160) -> (64, 80, 80)
                    "C2f_4": C2f(256, 256, n=2, shortcut=True),  # 4 (P3) (64, 80, 80)
                    "Conv_5": Conv(256, 512, 3, 2),  # 5-P4/16 (64, 80, 80) -> (128, 40, 40)
                    "C2f_6": C2f(512, 512, n=3, shortcut=True),  # 6 (P4) (128, 40, 40)
                    "Conv_7": Conv(512, 512, 3, 2),  # 7-P5/32 (128, 40, 40) -> (256, 20, 20)
                    "C2f_8": C2f(512, 512, n=1, shortcut=True),  # 8 (P5) (256, 20, 20)
                    "SPPF_9": SPPF(512, 512, k=5),  # 9 (P5) (256, 20, 20)
                }
            )
            # ir backbone
            self.ir_backbone = nn.ModuleDict(
                {
                    "Conv_0": Conv(self.channels, 16, 6, 2, 2),  # 10-P1/2 (3, 640, 640) -> (16, 320, 320)
                    "Conv_1": Conv(16, 32, 3, 2),  # 11-P2/4 (16, 320, 320) -> (32, 160, 160)
                    "C2f_2": C2f(32, 32, n=1, shortcut=True),  # 12 (P2) (32, 160, 160)
                    "Conv_3": Conv(32, 64, 3, 2),  # 13-P3/8 (32, 160, 160) -> (64, 80, 80)
                    "C2f_4": C2f(64, 64, n=2, shortcut=True),  # 14 (P3) (64, 80, 80)
                    "Conv_5": Conv(64, 128, 3, 2),  # 15-P4/16 (64, 80, 80) -> (128, 40, 40)
                    "C2f_6": C2f(128, 128, n=3, shortcut=True),  # 16 (P4) (128, 40, 40)
                    "Conv_7": Conv(128, 256, 3, 2),  # 17-P5/32 (128, 40, 40) -> (256, 20, 20)
                    "C2f_8": C2f(256, 256, n=1, shortcut=True),  # 18 (P5) (256, 20, 20)
                    "SPPF_9": SPPF(256, 256, k=5),  # 19 (P5) (256, 20, 20)
                }
            )
            # neck
            self.neck = nn.ModuleDict(
                {
                    # multi modal fusion
                    "FusionMamba": FusionMamba(256, 8, 8),  # 20: from [P5_img, P5_ir]
                    "FourInputFusionBlock": FourInputFusionBlock(256),  # 21
                    # common ops
                    "Concat": Concat(1),
                    "Upsample": nn.Upsample(scale_factor=2, mode="nearest"),
                    # FPN / PAN
                    "Conv_1": Conv(128, 128, 1, 1),
                    "C2f_1": C2f(128 * 3, 128, n=1, shortcut=False),
                    "Conv_2": Conv(128, 64, 1, 1),
                    "C2f_2": C2f(64 * 3, 64, n=1, shortcut=False),
                    "Conv_3": Conv(64, 64, 3, 2),
                    "C2f_3": C2f(64 + 64, 128, n=1, shortcut=False),
                    "Conv_4": Conv(128, 128, 3, 2),
                    "C2f_4": C2f(128 + 256, 256, n=1, shortcut=False),
                }
            )
            # head
            self.head = DetectV8(nc=self.nc, ch=(256, 512, 1024))

        else:
            # img backbone
            self.img_backbone = nn.ModuleDict(
                {
                    "Conv_0": Conv(self.channels, 16, 6, 2, 2),  # 0-P1/2 (3, 640, 640) -> (16, 320, 320)
                    "Conv_1": Conv(16, 32, 3, 2),  # 1-P2/4 (16, 320, 320) -> (32, 160, 160)
                    "C2f_2": C2f(32, 32, n=1, shortcut=True),  # 2 (P2) (32, 160, 160)
                    "Conv_3": Conv(32, 64, 3, 2),  # 3-P3/8 (32, 160, 160) -> (64, 80, 80)
                    "C2f_4": C2f(64, 64, n=2, shortcut=True),  # 4 (P3) (64, 80, 80)
                    "Conv_5": Conv(64, 128, 3, 2),  # 5-P4/16 (64, 80, 80) -> (128, 40, 40)
                    "C2f_6": C2f(128, 128, n=3, shortcut=True),  # 6 (P4) (128, 40, 40)
                    "Conv_7": Conv(128, 256, 3, 2),  # 7-P5/32 (128, 40, 40) -> (256, 20, 20)
                    "C2f_8": C2f(256, 256, n=1, shortcut=True),  # 8 (P5) (256, 20, 20)
                    "SPPF_9": SPPF(256, 256, k=5),  # 9 (P5) (256, 20, 20)
                }
            )
            # ir backbone
            self.ir_backbone = nn.ModuleDict(
                {
                    "Conv_0": Conv(self.channels, 16, 6, 2, 2),  # 10-P1/2 (3, 640, 640) -> (16, 320, 320)
                    "Conv_1": Conv(16, 32, 3, 2),  # 11-P2/4 (16, 320, 320) -> (32, 160, 160)
                    "C2f_2": C2f(32, 32, n=1, shortcut=True),  # 12 (P2) (32, 160, 160)
                    "Conv_3": Conv(32, 64, 3, 2),  # 13-P3/8 (32, 160, 160) -> (64, 80, 80)
                    "C2f_4": C2f(64, 64, n=2, shortcut=True),  # 14 (P3) (64, 80, 80)
                    "Conv_5": Conv(64, 128, 3, 2),  # 15-P4/16 (64, 80, 80) -> (128, 40, 40)
                    "C2f_6": C2f(128, 128, n=3, shortcut=True),  # 16 (P4) (128, 40, 40)
                    "Conv_7": Conv(128, 256, 3, 2),  # 17-P5/32 (128, 40, 40) -> (256, 20, 20)
                    "C2f_8": C2f(256, 256, n=1, shortcut=True),  # 18 (P5) (256, 20, 20)
                    "SPPF_9": SPPF(256, 256, k=5),  # 19 (P5) (256, 20, 20)
                }
            )
            # neck
            self.neck = nn.ModuleDict(
                {
                    # multi modal fusion
                    "FusionMamba": FusionMamba(256, 8, 8),  # 20: from [P5_img, P5_ir]
                    "FourInputFusionBlock": FourInputFusionBlock(256),  # 21
                    # common ops
                    "Concat": Concat(1),
                    "Upsample": nn.Upsample(scale_factor=2, mode="nearest"),
                    # FPN
                    "Conv_1": Conv(128, 128, 1, 1),
                    "C2f_1": C2f(128 * 3, 128, n=1, shortcut=False),
                    "Conv_2": Conv(128, 64, 1, 1),
                    "C2f_2": C2f(64 * 3, 64, n=1, shortcut=False),
                    # PAN
                    "Conv_3": Conv(64, 64, 3, 2),
                    "C2f_3": C2f(64 + 64, 128, n=1, shortcut=False),
                    "Conv_4": Conv(128, 128, 3, 2),
                    "C2f_4": C2f(128 + 256, 256, n=1, shortcut=False),
                }
            )
            # head
            self.head = DetectV8(nc=self.nc, ch=(256, 512, 1024))

    @property
    def dummy_input(self) -> tuple[torch.Tensor]:
        dummy_img_input = torch.randn(1, self.channels, self.imgsz, self.imgsz, device=self.device)
        dummy_ir_input = torch.randn(1, self.channels, self.imgsz, self.imgsz, device=self.device)
        return dummy_img_input, dummy_ir_input

    def _rgb_forward_backbone(self, x: torch.Tensor):
        x0 = self.img_backbone["Conv_0"](x)
        x1 = self.img_backbone["Conv_1"](x0)
        x2 = self.img_backbone["C2f_2"](x1)
        x3 = self.img_backbone["Conv_3"](x2)
        x4 = self.img_backbone["C2f_4"](x3)  # P3_rgb
        x5 = self.img_backbone["Conv_5"](x4)
        x6 = self.img_backbone["C2f_6"](x5)  # P4_rgb
        x7 = self.img_backbone["Conv_7"](x6)
        x8 = self.img_backbone["C2f_8"](x7)
        x9 = self.img_backbone["SPPF_9"](x8)  # P5_rgb
        return x4, x6, x9

    def _ir_forward_backbone(self, x: torch.Tensor):
        x0 = self.ir_backbone["Conv_0"](x)
        x1 = self.ir_backbone["Conv_1"](x0)
        x2 = self.ir_backbone["C2f_2"](x1)
        x3 = self.ir_backbone["Conv_3"](x2)
        x4 = self.ir_backbone["C2f_4"](x3)  # P3_ir
        x5 = self.ir_backbone["Conv_5"](x4)
        x6 = self.ir_backbone["C2f_6"](x5)  # P4_ir
        x7 = self.ir_backbone["Conv_7"](x6)
        x8 = self.ir_backbone["C2f_8"](x7)
        x9 = self.ir_backbone["SPPF_9"](x8)  # P5_ir
        return x4, x6, x9

    def _initialize_weights(self):
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
            [self.imgsz / x.shape[-2] for x in self.forward(*self.dummy_input)], dtype=torch.int8, device=self.device
        )
        self.head.stride = self.stride

    def forward(self, imgs: torch.Tensor, irs: torch.Tensor) -> tuple[torch.Tensor]:
        # The channel annotation takes "n" net scale as an example
        # backbone
        p3_rgb, p4_rgb, p5_rgb = self._rgb_forward_backbone(imgs)  # (64, 128, 256)
        p3_ir, p4_ir, p5_ir = self._ir_forward_backbone(irs)  # (64, 128, 256)

        # multi modal fusion
        p5_fused = self.neck["FusionMamba"]([p5_rgb, p5_ir])  # 256 -> 256
        p5_fuse_block = self.neck["FourInputFusionBlock"]([p5_fused, p5_rgb, p5_ir])  # 256 -> 256

        # FPN
        p5_top = self.neck["Conv_fuse_p5"](p5_fuse_block)  # 256 -> 128
        up_p5 = self.neck["Upsample"](p5_top)  # 128 -> 128
        p4_cat = self.neck["Concat"]([up_p5, p4_rgb, p4_ir])  # 128 * 3
        p4_out = self.neck["C2f_p4"](p4_cat)  # 128 * 3 -> 128

        p4_red = self.neck["Conv_p4_to_p3"](p4_out)  # 128 -> 64
        up_p4 = self.neck["Upsample"](p4_red)  # 64 -> 64
        p3_cat = self.neck["Concat"]([up_p4, p3_rgb, p3_ir])  # 64 * 3
        p3_out = self.neck["C2f_p3"](p3_cat)  # 64 * 3 -> 64

        # PAN
        p3_down = self.neck["Conv_p3_down"](p3_out)  # 64 -> 64
        p4_pan_cat = self.neck["Concat"]([p3_down, p4_red])  # 64 + 64
        p4_pan = self.neck["C2f_pan_p4"](p4_pan_cat)  # 64 + 128 -> 128

        p4_down = self.neck["Conv_p4_down"](p4_pan)  # 128 -> 128
        p5_pan_cat = self.neck["Concat"]([p4_down, p5_fuse_block])  # 128 + 256
        p5_pan = self.neck["C2f_pan_p5"](p5_pan_cat)  # 128 + 256 -> 256

        # Detect head: [P3, P4, P5]
        return self.head([p3_out, p4_pan, p5_pan])
