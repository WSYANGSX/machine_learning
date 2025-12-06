from typing import Literal

import torch
import torch.nn as nn

from machine_learning.networks import BaseNet
from machine_learning.modules.heads import DetectV8
from machine_learning.modules.blocks import CHyperACE, MMFullPAD_Tunnel
from ultralytics.nn.modules import Conv, DSC3k2, DSConv, A2C2f, HyperACE, DownsampleConv, Concat, FullPAD_Tunnel


class MMICNet(BaseNet):
    def __init__(
        self,
        imgsz: int,
        channels: int = 3,
        nc: int = 1,
        net_scale: Literal["n", "s", "l", "x"] = "n",
        *args,
        **kwargs,
    ):
        """Multimodal object detection network.

        Args:
            img_shape (Sequence[int]): the shape of input rgb image.
            ir_shape (Sequence[int]): the shape of input ir image.
            num_classes (int): number of classes.
        """
        super().__init__(args=args, kwargs=kwargs)

        self.imgsz = imgsz
        self.nc = nc
        self.net_scale = net_scale
        self.channels = channels

        if self.net_scale == "n":
            # img backbone
            self.img_backbone = nn.ModuleDict(
                {
                    "Conv_1": Conv(self.channels, 16, 3, 2),  # 0-P1/2  (3, 640, 640) -> (16, 320, 320)
                    "Conv_2": Conv(16, 32, 3, 2, 1, 2),  # 1-P2/4 (16, 320, 320) -> (32, 160, 160)
                    "DSC3k2_1": DSC3k2(32, 64, 1, False, 0.25),  # 2 (P2) (32, 160, 160) -> (64, 160, 160)
                    "Conv_3": Conv(64, 64, 3, 2, 1, 4),  # 3-P3/8 (64, 160, 160) -> (64, 80, 80)
                    "DSC3k2_2": DSC3k2(64, 128, 1, False, 0.25),  # 4 (P3) (64, 80, 80) -> (128, 80, 80)
                    "DSConv_1": DSConv(128, 128, 3, 2),  # 5-P4/16 (128, 80, 80) -> (128, 40, 40)
                    "A2C2f_1": A2C2f(128, 128, 2, True, 4),  # 6 (P4) (128, 40, 40)
                    "DSConv_2": DSConv(128, 256, 3, 2),  # 7-P5/32 (128, 40, 40) -> (128, 20, 20)
                    "A2C2f_2": A2C2f(256, 256, 2, True, 1),  # 8 (P5) (256, 20, 20)
                }
            )
            # ir backbone
            self.ir_backbone = nn.ModuleDict(
                {
                    "Conv_1": Conv(self.channels, 16, 3, 2),  # 0-P1/2  (3, 640, 640) -> (16, 320, 320)
                    "Conv_2": Conv(16, 32, 3, 2, 1, 2),  # 1-P2/4 (16, 320, 320) -> (32, 160, 160)
                    "DSC3k2_1": DSC3k2(32, 64, 1, False, 0.25),  # 2 (P2) (32, 160, 160) -> (64, 160, 160)
                    "Conv_3": Conv(64, 64, 3, 2, 1, 4),  # 3-P3/8 (64, 160, 160) -> (64, 80, 80)
                    "DSC3k2_2": DSC3k2(64, 128, 1, False, 0.25),  # 4 (P3) (64, 80, 80) -> (128, 80, 80)
                    "DSConv_1": DSConv(128, 128, 3, 2),  # 5-P4/16 (128, 80, 80) -> (128, 40, 40)
                    "A2C2f_1": A2C2f(128, 128, 2, True, 4),  # 6 (P4) (128, 40, 40)
                    "DSConv_2": DSConv(128, 256, 3, 2),  # 7-P5/32 (128, 40, 40) -> (128, 20, 20)
                    "A2C2f_2": A2C2f(256, 256, 2, True, 1),  # 8 (P5) (256, 20, 20)
                }
            )
            # neck
            self.neck = nn.ModuleDict(
                {
                    # uable to train
                    "Upsample": nn.Upsample(None, 2, "nearest"),
                    "Cat": Concat(1),
                    # able to train
                    "HyperACE_Img": HyperACE(128, 128, 2, 8, True, True, 0.5, 1, "both"),
                    "Downsample_1": DownsampleConv(128),
                    "HyperACE_Ir": HyperACE(128, 128, 2, 8, True, True, 0.5, 1, "both"),
                    "Downsample_2": DownsampleConv(128),
                    "CHyperACE": CHyperACE(128, 128, 2, 8, True, True, 0.5, 1, "both"),
                    "Downsample_3": DownsampleConv(128),
                    "FullPAD_Tunnel_1": FullPAD_Tunnel(),
                    "FullPAD_Tunnel_2": FullPAD_Tunnel(),
                    "FullPAD_Tunnel_3": FullPAD_Tunnel(),
                    "DSC3k2_1": DSC3k2(384, 128, 1, False),
                    "MMFullPAD_Tunnel_4": MMFullPAD_Tunnel(),
                    "DSC3k2_2": DSC3k2(256, 64, 1, False),
                    "Conv_1_1": Conv(128, 64, 1, 1),
                    "Conv_1_2": Conv(128, 64, 1, 1),
                    "Conv_1_3": Conv(128, 64, 1, 1),
                    "MMFullPAD_Tunnel_5": MMFullPAD_Tunnel(),
                    "Conv_2": Conv(64, 64, 3, 2),
                    "DSC3k2_3": DSC3k2(192, 128, 1, False),
                    "MMFullPAD_Tunnel_6": MMFullPAD_Tunnel(),
                    "Conv_3": Conv(128, 128, 3, 2),
                    "DSC3k2_4": DSC3k2(384, 256, 1, False),
                    "MMFullPAD_Tunnel_7": MMFullPAD_Tunnel(),
                }
            )
            # head
            self.head = DetectV8(nc=self.nc, ch=(64, 128, 256))

        elif self.net_scale == "s":
            # img backbone
            self.img_backbone = nn.ModuleDict(
                {
                    "Conv_1": Conv(self.channels, 32, 3, 2),  # 0 P1/2 (3, 640, 640) -> (32, 320, 320)
                    "Conv_2": Conv(32, 64, 3, 2, 1, 2),  # 1 P2/4 (32, 320, 320) -> (64, 160, 160)
                    "DSC3k2_1": DSC3k2(64, 128, 1, False, 0.25),  # 2 (P2) (64, 160, 160) -> (128, 160, 160)
                    "Conv_3": Conv(128, 128, 3, 2, 1, 4),  # 3 P3/8 (128, 160, 160) -> (128, 80, 80)
                    "DSC3k2_2": DSC3k2(128, 256, 1, False, 0.25),  # 4 (P3) (128, 80, 80) -> (256, 80, 80)
                    "DSConv_1": DSConv(256, 256, 3, 2),  # 5 P4/16 (256, 80, 80) -> (256, 40, 40)
                    "A2C2f_1": A2C2f(256, 256, 2, True, 4),  # 6 (P4) (256, 40, 40)
                    "DSConv_2": DSConv(256, 512, 3, 2),  # 7 P5/32 (256, 40, 40) -> (512, 20, 20)
                    "A2C2f_2": A2C2f(512, 512, 2, True, 1),  # 8 (P5) (512, 20, 20)
                }
            )
            # ir backbone
            self.ir_backbone = nn.ModuleDict(
                {
                    "Conv_1": Conv(self.channels, 32, 3, 2),  # 0 P1/2 (3, 640, 640) -> (32, 320, 320)
                    "Conv_2": Conv(32, 64, 3, 2, 1, 2),  # 1 P2/4 (32, 320, 320) -> (64, 160, 160)
                    "DSC3k2_1": DSC3k2(64, 128, 1, False, 0.25),  # 2 (P2) (64, 160, 160) -> (128, 160, 160)
                    "Conv_3": Conv(128, 128, 3, 2, 1, 4),  # 3 P3/8 (128, 160, 160) -> (128, 80, 80)
                    "DSC3k2_2": DSC3k2(128, 256, 1, False, 0.25),  # 4 (P3) (128, 80, 80) -> (256, 80, 80)
                    "DSConv_1": DSConv(256, 256, 3, 2),  # 5 P4/16 (256, 80, 80) -> (256, 40, 40)
                    "A2C2f_1": A2C2f(256, 256, 2, True, 4),  # 6 (P4) (256, 40, 40)
                    "DSConv_2": DSConv(256, 512, 3, 2),  # 7 P5/32 (256, 40, 40) -> (512, 20, 20)
                    "A2C2f_2": A2C2f(512, 512, 2, True, 1),  # 8 (P5) (512, 20, 20)
                }
            )
            # neck
            self.neck = nn.ModuleDict(
                {
                    # uable to train
                    "Upsample": nn.Upsample(None, 2, "nearest"),
                    "Cat": Concat(1),
                    # able to train
                    "HyperACE_Img": HyperACE(256, 256, 1, 8, True, True, 0.5, 1, "both"),
                    "Downsample_1": DownsampleConv(256),
                    "HyperACE_Ir": HyperACE(256, 256, 1, 8, True, True, 0.5, 1, "both"),
                    "Downsample_2": DownsampleConv(256),
                    "CHyperACE": CHyperACE(256, 256, 1, 8, True, True, 0.5, 1, "both"),
                    "Downsample_3": DownsampleConv(256),
                    "FullPAD_Tunnel_1": FullPAD_Tunnel(),
                    "FullPAD_Tunnel_2": FullPAD_Tunnel(),
                    "FullPAD_Tunnel_3": FullPAD_Tunnel(),
                    "DSC3k2_1": DSC3k2(768, 256, 1, False),
                    "MMFullPAD_Tunnel_4": MMFullPAD_Tunnel(),
                    "DSC3k2_2": DSC3k2(512, 128, 1, False),
                    "Conv_1_1": Conv(256, 128, 1, 1),
                    "Conv_1_2": Conv(256, 128, 1, 1),
                    "Conv_1_3": Conv(256, 128, 1, 1),
                    "MMFullPAD_Tunnel_5": MMFullPAD_Tunnel(),
                    "Conv_2": Conv(128, 128, 3, 2),
                    "DSC3k2_3": DSC3k2(384, 256, 1, False),
                    "MMFullPAD_Tunnel_6": MMFullPAD_Tunnel(),
                    "Conv_3": Conv(256, 256, 3, 2),
                    "DSC3k2_4": DSC3k2(768, 512, 1, False),
                    "MMFullPAD_Tunnel_7": MMFullPAD_Tunnel(),
                }
            )
            # head
            self.head = DetectV8(nc=self.nc, ch=(128, 256, 512))

        elif self.net_scale == "l":
            # img backbone
            self.img_backbone = nn.ModuleDict(
                {
                    "Conv_1": Conv(self.channels, 64, 3, 2),  # 0 P1/2 (3, 640, 640) -> (64, 320, 320)
                    "Conv_2": Conv(64, 128, 3, 2, 1, 2),  # 1 P2/4 (64, 320, 320) -> (128, 160, 160)
                    "DSC3k2_1": DSC3k2(128, 256, 2, True, 0.25),  # 2 (P2) (128, 160, 160) -> (256, 160, 160)
                    "Conv_3": Conv(256, 256, 3, 2, 1, 4),  # 3 P3/8 (256, 160, 160) -> (256, 80, 80)
                    "DSC3k2_2": DSC3k2(256, 512, 2, True, 0.25),  # 4 (P3) (256, 80, 80) -> (512, 80, 80)
                    "DSConv_1": DSConv(512, 512, 3, 2),  # 5 P4/16 (512, 80, 80) -> (512, 40, 40)
                    "A2C2f_1": A2C2f(512, 512, 4, True, 4, True, 1.5),  # 6 (P4) (512, 40, 40)
                    "DSConv_2": DSConv(512, 512, 3, 2),  # 7 P5/32 (512, 40, 40) -> (512, 20, 20)
                    "A2C2f_2": A2C2f(512, 512, 4, True, 1, True, 1.5),  # 8 (P5) (512, 20, 20)
                }
            )
            # ir backbone
            self.ir_backbone = nn.ModuleDict(
                {
                    "Conv_1": Conv(self.channels, 64, 3, 2),  # 0 P1/2 (3, 640, 640) -> (64, 320, 320)
                    "Conv_2": Conv(64, 128, 3, 2, 1, 2),  # 1 P2/4 (64, 320, 320) -> (128, 160, 160)
                    "DSC3k2_1": DSC3k2(128, 256, 2, True, 0.25),  # 2 (P2) (128, 160, 160) -> (256, 160, 160)
                    "Conv_3": Conv(256, 256, 3, 2, 1, 4),  # 3 P3/8 (256, 160, 160) -> (256, 80, 80)
                    "DSC3k2_2": DSC3k2(256, 512, 2, True, 0.25),  # 4 (P3) (256, 80, 80) -> (512, 80, 80)
                    "DSConv_1": DSConv(512, 512, 3, 2),  # 5 P4/16 (512, 80, 80) -> (512, 40, 40)
                    "A2C2f_1": A2C2f(512, 512, 4, True, 4, True, 1.5),  # 6 (P4) (512, 40, 40)
                    "DSConv_2": DSConv(512, 512, 3, 2),  # 7 P5/32 (512, 40, 40) -> (512, 20, 20)
                    "A2C2f_2": A2C2f(512, 512, 4, True, 1, True, 1.5),  # 8 (P5) (512, 20, 20)
                }
            )
            # neck
            self.neck = nn.ModuleDict(
                {
                    # uable to train
                    "Upsample": nn.Upsample(None, 2, "nearest"),
                    "Cat": Concat(1),
                    # able to train
                    "HyperACE_Img": HyperACE(512, 512, 2, 8, True, True, 0.5, 1, "both", False),
                    "Downsample_1": DownsampleConv(512, False),
                    "HyperACE_Ir": HyperACE(512, 512, 2, 8, True, True, 0.5, 1, "both", False),
                    "Downsample_2": DownsampleConv(512, False),
                    "CHyperACE": CHyperACE(512, 512, 2, 8, True, True, 0.5, 1, "both"),
                    "Downsample_3": DownsampleConv(512, False),
                    "FullPAD_Tunnel_1": FullPAD_Tunnel(),
                    "FullPAD_Tunnel_2": FullPAD_Tunnel(),
                    "FullPAD_Tunnel_3": FullPAD_Tunnel(),
                    "DSC3k2_1": DSC3k2(1024, 512, 2, True),
                    "MMFullPAD_Tunnel_4": MMFullPAD_Tunnel(),
                    "DSC3k2_2": DSC3k2(1024, 256, 2, True),
                    "Conv_1_1": Conv(512, 256, 1, 1),
                    "Conv_1_2": Conv(512, 256, 1, 1),
                    "Conv_1_3": Conv(512, 256, 1, 1),
                    "MMFullPAD_Tunnel_5": MMFullPAD_Tunnel(),
                    "Conv_2": Conv(256, 256, 3, 2),
                    "DSC3k2_3": DSC3k2(768, 512, 2, True),
                    "MMFullPAD_Tunnel_6": MMFullPAD_Tunnel(),
                    "Conv_3": Conv(512, 512, 3, 2),
                    "DSC3k2_4": DSC3k2(1024, 512, 2, True),
                    "MMFullPAD_Tunnel_7": MMFullPAD_Tunnel(),
                }
            )
            # head
            self.head = DetectV8(nc=self.nc, ch=(256, 512, 512))

        else:
            # img backbone
            self.img_backbone = nn.ModuleDict(
                {
                    "Conv_1": Conv(self.channels, 96, 3, 2),  # 0 P1/2  (3, 640, 640) -> (96, 320, 320)
                    "Conv_2": Conv(96, 192, 3, 2, 1, 2),  # 1 P2/4 (96, 320, 320) -> (192, 160, 160)
                    "DSC3k2_1": DSC3k2(192, 384, 2, True, 0.25),  # 2 (P2) (192, 160, 160) -> (384, 160, 160)
                    "Conv_3": Conv(384, 384, 3, 2, 1, 4),  # 3 P3/8 (384, 160, 160) -> (384, 80, 80)
                    "DSC3k2_2": DSC3k2(384, 768, 2, True, 0.25),  # 4 (P3) (384, 80, 80) -> (768, 80, 80)
                    "DSConv_1": DSConv(768, 768, 3, 2),  # 5 P4/16 (768, 80, 80) -> (768, 40, 40)
                    "A2C2f_1": A2C2f(768, 768, 4, True, 4, True, 1.5),  # 6 (P4) (768, 40, 40)
                    "DSConv_2": DSConv(768, 768, 3, 2),  # 7 P5/32 (768, 40, 40) -> (768, 20, 20)
                    "A2C2f_2": A2C2f(768, 768, 4, True, 1, True, 1.5),  # 8 (P5) (768, 20, 20)
                }
            )
            # ir backbone
            self.ir_backbone = nn.ModuleDict(
                {
                    "Conv_1": Conv(self.channels, 96, 3, 2),  # 0 P1/2  (3, 640, 640) -> (96, 320, 320)
                    "Conv_2": Conv(96, 192, 3, 2, 1, 2),  # 1 P2/4 (96, 320, 320) -> (192, 160, 160)
                    "DSC3k2_1": DSC3k2(192, 384, 2, True, 0.25),  # 2 (P2) (192, 160, 160) -> (384, 160, 160)
                    "Conv_3": Conv(384, 384, 3, 2, 1, 4),  # 3 P3/8 (384, 160, 160) -> (384, 80, 80)
                    "DSC3k2_2": DSC3k2(384, 768, 2, True, 0.25),  # 4 (P3) (384, 80, 80) -> (768, 80, 80)
                    "DSConv_1": DSConv(768, 768, 3, 2),  # 5 P4/16 (768, 80, 80) -> (768, 40, 40)
                    "A2C2f_1": A2C2f(768, 768, 4, True, 4, True, 1.5),  # 6 (P4) (768, 40, 40)
                    "DSConv_2": DSConv(768, 768, 3, 2),  # 7 P5/32 (768, 40, 40) -> (768, 20, 20)
                    "A2C2f_2": A2C2f(768, 768, 4, True, 1, True, 1.5),  # 8 (P5) (768, 20, 20)
                }
            )
            # neck
            self.neck = nn.ModuleDict(
                {
                    # uable to train
                    "Upsample": nn.Upsample(None, 2, "nearest"),
                    "Cat": Concat(1),
                    # able to train
                    "HyperACE_Img": HyperACE(768, 768, 2, 12, True, True, 0.5, 1, "both", False),
                    "Downsample_1": DownsampleConv(768, False),
                    "HyperACE_Ir": HyperACE(768, 768, 2, 12, True, True, 0.5, 1, "both", False),
                    "Downsample_2": DownsampleConv(768, False),
                    "CHyperACE": CHyperACE(768, 768, 2, 12, True, True, 0.5, 1, "both"),
                    "Downsample_3": DownsampleConv(768, False),
                    "FullPAD_Tunnel_1": FullPAD_Tunnel(),
                    "FullPAD_Tunnel_2": FullPAD_Tunnel(),
                    "FullPAD_Tunnel_3": FullPAD_Tunnel(),
                    "DSC3k2_1": DSC3k2(1536, 768, 1, True),
                    "MMFullPAD_Tunnel_4": MMFullPAD_Tunnel(),
                    "DSC3k2_2": DSC3k2(1536, 384, 1, True),
                    "Conv_1_1": Conv(768, 384, 1, 1),
                    "Conv_1_2": Conv(768, 384, 1, 1),
                    "Conv_1_3": Conv(768, 384, 1, 1),
                    "MMFullPAD_Tunnel_5": MMFullPAD_Tunnel(),
                    "Conv_2": Conv(384, 384, 3, 2),
                    "DSC3k2_3": DSC3k2(1152, 768, 1, True),
                    "MMFullPAD_Tunnel_6": MMFullPAD_Tunnel(),
                    "Conv_3": Conv(768, 768, 3, 2),
                    "DSC3k2_4": DSC3k2(1536, 768, 1, True),
                    "MMFullPAD_Tunnel_7": MMFullPAD_Tunnel(),
                }
            )
            # head
            self.head = DetectV8(nc=self.nc, ch=(256, 512, 512))

    def forward(self, imgs: torch.Tensor, irs: torch.Tensor) -> tuple[torch.Tensor]:
        # img backbone
        img_skips = []
        for key, layer in self.img_backbone.items():
            imgs = layer(imgs)
            if key in ["DSC3k2_2", "A2C2f_1", "A2C2f_2"]:
                img_skips.append(imgs)

        # ir backbone
        ir_skips = []
        for key, layer in self.ir_backbone.items():
            irs = layer(irs)
            if key in ["DSC3k2_2", "A2C2f_1", "A2C2f_2"]:
                ir_skips.append(irs)

        pixels_fuse = [img_skips[i] + ir_skips[i] for i in range(len(img_skips))]

        # ----- neck -----
        # Inter HyperACE
        img_enhanced = []
        ir_enhanced = []

        img_h2 = self.neck.HyperACE_Img(img_skips)
        ir_h2 = self.neck.HyperACE_Ir(ir_skips)

        img_h1 = self.neck.Upsample(img_h2)
        ir_h1 = self.neck.Upsample(ir_h2)

        img_enhanced.append(img_h1)
        ir_enhanced.append(ir_h1)
        img_enhanced.append(img_h2)
        ir_enhanced.append(ir_h2)

        img_h3 = self.neck.Downsample_1(img_h2)
        ir_h3 = self.neck.Downsample_3(ir_h2)

        img_enhanced.append(img_h3)
        ir_enhanced.append(ir_h3)

        # Cross HyerACE
        fuse_enhanced = []
        fuse_h2 = self.neck.CHyperACE([img_enhanced[1], ir_enhanced[1]])
        fuse_h1 = self.neck.Upsample(fuse_h2)
        fuse_h3 = self.neck.Downsample_3(fuse_h2)
        fuse_enhanced.append(fuse_h1)
        fuse_enhanced.append(fuse_h2)
        fuse_enhanced.append(fuse_h3)

        f1 = self.neck.FullPAD_Tunnel_1([pixels_fuse[0], fuse_enhanced[0]])
        f2 = self.neck.FullPAD_Tunnel_2([pixels_fuse[1], fuse_enhanced[1]])
        f3 = self.neck.FullPAD_Tunnel_3([pixels_fuse[2], fuse_enhanced[2]])

        # Full_Tunnel
        d1 = self.neck.DSC3k2_1(self.neck.Cat([self.neck.Upsample(f3), f2]))
        f5 = self.neck.MMFullPAD_Tunnel_5([d1, img_enhanced[1], ir_enhanced[1], fuse_enhanced[1]])
        d2 = self.neck.DSC3k2_2(self.neck.Cat([self.neck.Upsample(d1), f1]))
        # det1
        det1 = f4 = self.neck.MMFullPAD_Tunnel_4(
            [
                d2,
                self.neck.Conv_1_1(img_enhanced[0]),  # 是否分开Conv？
                self.neck.Conv_1_2(ir_enhanced[0]),
                self.neck.Conv_1_3(fuse_enhanced[0]),
            ]
        )
        d3 = self.neck.DSC3k2_3(self.neck.Cat([self.neck.Conv_2(f4), f5]))
        # det2
        det2 = self.neck.MMFullPAD_Tunnel_6([d3, img_enhanced[1], ir_enhanced[1], fuse_enhanced[1]])
        # det3
        d4 = self.neck.DSC3k2_4(self.neck.Cat([self.neck.Conv_3(d3), f3]))
        det3 = self.neck.MMFullPAD_Tunnel_7([d4, img_enhanced[2], ir_enhanced[2], fuse_enhanced[2]])

        return self.head([det1, det2, det3])

    def view_structure(self) -> None:
        super().view_structure()

        from torchinfo import summary

        img_input = torch.randn(1, self.channels, self.imgsz, self.imgsz, device=self.device)
        ir_input = torch.randn(1, self.channels, self.imgsz, self.imgsz, device=self.device)

        summary(self, input_data=[img_input, ir_input])

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
        img_input = torch.randn(1, self.channels, self.imgsz, self.imgsz, device=self.device)
        ir_input = torch.randn(1, self.channels, self.imgsz, self.imgsz, device=self.device)
        self.stride = torch.tensor(
            [self.imgsz / x.shape[-2] for x in self.forward(img_input, ir_input)], dtype=torch.int8, device=self.device
        )
        self.head.stride = self.stride
