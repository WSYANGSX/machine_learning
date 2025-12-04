from typing import Literal

import torch
import torch.nn as nn

from machine_learning.networks import BaseNet
from machine_learning.modules.head import DetectV8
from machine_learning.modules.blocks import FusionMamba, FourInputFusionBlock

from ultralytics.nn.modules import Conv, Concat, C2f, SPPF


class COMONet(BaseNet):
    def __init__(
        self,
        imgsz: int,
        channels: int = 3,
        nc: int = 1,
        net_scale: Literal["n", "s", "l", "x"] = "n",
        *args,
        **kwargs,
    ):
        """
        COMONet: multimodal object detection network (RGB + IR).

        Args:
            imgsz (int): Input the image size.
            channels (int): The number of input channels for each mode.
            nc (int): number of classes.
        """
        super().__init__(args=args, kwargs=kwargs)

        self.imgsz = imgsz
        self.nc = nc
        self.channels = channels
        self.net_scale = net_scale

        # ---------------- RGB backbone（对应 backbone 0-9） ----------------
        self.rgb_backbone = nn.ModuleDict(
            {
                "Conv_0": Conv(self.channels, 64, 6, 2, 2),  # 0-P1/2
                "Conv_1": Conv(64, 128, 3, 2),  # 1-P2/4
                "C2f_2": C2f(128, 128, n=3, shortcut=True),  # 2
                "Conv_3": Conv(128, 256, 3, 2),  # 3-P3/8
                "C2f_4": C2f(256, 256, n=6, shortcut=True),  # 4 (P3)
                "Conv_5": Conv(256, 512, 3, 2),  # 5-P4/16
                "C2f_6": C2f(512, 512, n=9, shortcut=True),  # 6 (P4)
                "Conv_7": Conv(512, 1024, 3, 2),  # 7-P5/32
                "C2f_8": C2f(1024, 1024, n=3, shortcut=True),  # 8
                "SPPF_9": SPPF(1024, 1024, k=5),  # 9 (P5)
            }
        )

        # ---------------- IR backbone（对应 backbone 10-19） ----------------
        # 这里我做成与 RGB 对称的一套结构，方便你接单通道 IR
        self.ir_backbone = nn.ModuleDict(
            {
                "Conv_10": Conv(self.channels, 64, 6, 2, 2),  # 10-P1/2
                "Conv_11": Conv(64, 128, 3, 2),  # 11-P2/4
                "C2f_12": C2f(128, 128, n=3, shortcut=True),  # 12
                "Conv_13": Conv(128, 256, 3, 2),  # 13-P3/8
                "C2f_14": C2f(256, 256, n=6, shortcut=True),  # 14 (P3)
                "Conv_15": Conv(256, 512, 3, 2),  # 15-P4/16
                "C2f_16": C2f(512, 512, n=9, shortcut=True),  # 16 (P4)
                "Conv_17": Conv(512, 1024, 3, 2),  # 17-P5/32
                "C2f_18": C2f(1024, 1024, n=3, shortcut=True),  # 18
                "SPPF_19": SPPF(1024, 1024, k=5),  # 19 (P5)
            }
        )

        # ---------------- neck + fusion + head ----------------
        self.neck = nn.ModuleDict(
            {
                # 多模态 P5 融合
                "FusionMamba": FusionMamba(1024, 8, 8),  # 20: from [P5_rgb, P5_ir]
                "FourInputFusionBlock": FourInputFusionBlock(512),  # 21: 输出 512 通道特征
                # 通用算子
                "Concat": Concat(1),
                "Upsample": nn.Upsample(scale_factor=2, mode="nearest"),
                # FPN / PAN 结构
                "Conv_fuse_p5": Conv(512, 512, 1, 1),  # 对 4IFB 输出做 1×1
                "C2f_p4": C2f(512 + 512 + 512, 512, n=3, shortcut=False),  # 上采样 P5 + P4_rgb + P4_ir
                "Conv_p4_to_p3": Conv(512, 256, 1, 1),  # P4→减通道
                "C2f_p3": C2f(256 + 256 + 256, 256, n=3, shortcut=False),  # 上采样 P4 + P3_rgb + P3_ir
                "Conv_p3_down": Conv(256, 256, 3, 2),  # P3 下采样到 P4 尺度
                "C2f_pan_p4": C2f(256 + 512, 512, n=3, shortcut=False),  # PAN P4
                "Conv_p4_down": Conv(512, 512, 3, 2),  # P4 下采样到 P5 尺度
                "C2f_pan_p5": C2f(512 + 512, 1024, n=3, shortcut=False),  # PAN P5
            }
        )

        # Detect head，通道顺序对应 [P3, P4, P5]
        self.head = DetectV8(nc=self.nc, ch=(256, 512, 1024))

        # 如果你需要 stride / anchor 初始化，这里可以按自己工程风格加上
        # self._initialize_strides()  # 如有需要可实现

    def _rgb_forward_backbone(self, x: torch.Tensor):
        x0 = self.rgb_backbone["Conv_0"](x)
        x1 = self.rgb_backbone["Conv_1"](x0)
        x2 = self.rgb_backbone["C2f_2"](x1)
        x3 = self.rgb_backbone["Conv_3"](x2)
        x4 = self.rgb_backbone["C2f_4"](x3)  # P3_rgb
        x5 = self.rgb_backbone["Conv_5"](x4)
        x6 = self.rgb_backbone["C2f_6"](x5)  # P4_rgb
        x7 = self.rgb_backbone["Conv_7"](x6)
        x8 = self.rgb_backbone["C2f_8"](x7)
        x9 = self.rgb_backbone["SPPF_9"](x8)  # P5_rgb
        return x4, x6, x9

    def _ir_forward_backbone(self, x: torch.Tensor):
        x10 = self.ir_backbone["Conv_10"](x)
        x11 = self.ir_backbone["Conv_11"](x10)
        x12 = self.ir_backbone["C2f_12"](x11)
        x13 = self.ir_backbone["Conv_13"](x12)
        x14 = self.ir_backbone["C2f_14"](x13)  # P3_ir
        x15 = self.ir_backbone["Conv_15"](x14)
        x16 = self.ir_backbone["C2f_16"](x15)  # P4_ir
        x17 = self.ir_backbone["Conv_17"](x16)
        x18 = self.ir_backbone["C2f_18"](x17)
        x19 = self.ir_backbone["SPPF_19"](x18)  # P5_ir
        return x14, x16, x19

    def forward(self, imgs: torch.Tensor, irs: torch.Tensor):
        # backbone
        p3_rgb, p4_rgb, p5_rgb = self._rgb_forward_backbone(imgs)
        p3_ir, p4_ir, p5_ir = self._ir_forward_backbone(irs)

        # P5 多模态融合
        p5_fused = self.neck["FusionMamba"]([p5_rgb, p5_ir])  # 20
        p5_fuse_block = self.neck["FourInputFusionBlock"]([p5_fused, p5_rgb, p5_ir])  # 21, 输出 512 通道

        # FPN: 自上而下
        p5_top = self.neck["Conv_fuse_p5"](p5_fuse_block)  # 512

        up_p5 = self.neck["Upsample"](p5_top)  # 上采样到 P4 尺度
        p4_cat = self.neck["Concat"]([up_p5, p4_rgb, p4_ir])
        p4_out = self.neck["C2f_p4"](p4_cat)  # 512

        p4_red = self.neck["Conv_p4_to_p3"](p4_out)  # 256
        up_p4 = self.neck["Upsample"](p4_red)
        p3_cat = self.neck["Concat"]([up_p4, p3_rgb, p3_ir])
        p3_out = self.neck["C2f_p3"](p3_cat)  # 256

        # PAN: 自下而上
        p3_down = self.neck["Conv_p3_down"](p3_out)  # -> P4 尺度
        p4_pan_cat = self.neck["Concat"]([p3_down, p4_out])
        p4_pan = self.neck["C2f_pan_p4"](p4_pan_cat)  # 512

        p4_down = self.neck["Conv_p4_down"](p4_pan)  # -> P5 尺度
        p5_pan_cat = self.neck["Concat"]([p4_down, p5_fuse_block])
        p5_pan = self.neck["C2f_pan_p5"](p5_pan_cat)  # 1024

        # Detect head: [P3, P4, P5]
        return self.head([p3_out, p4_pan, p5_pan])
