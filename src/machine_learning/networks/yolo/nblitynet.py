from __future__ import annotations
from typing import Sequence

import torch
import torch.nn as nn

from machine_learning.networks import BaseNet
from machine_learning.modules.head import DetectV8
from machine_learning.modules.blocks import CHyperACE, MMFullPAD_Tunnel

from ultralytics.nn.modules import Conv, DSC3k2, DSConv, A2C2f, HyperACE, DownsampleConv, Concat


class NblityNet(BaseNet):
    def __init__(
        self,
        img_shape: Sequence[int],
        thermal_shape: Sequence[int],
        nc: int = 1,
    ):
        """multimodal object detection network.

        Args:
            img_shape (Sequence[int]): the shape of input rgb image.
            thermal_shape (Sequence[int]): the shape of input thermal image.
            num_anchors (int): number of anchors.
            num_classes (int): number of classes.
        """
        super().__init__()
        self.img_shape = img_shape  # (3, height, width)
        self.thermal_shape = thermal_shape  # (1, height, width)
        self.nc = nc

        self.img_in_channels = self.img_shape[0]
        self.the_in_channels = self.thermal_shape[0]

        # backbones
        self.img_backbone = nn.ModuleDict(
            {
                "Conv_1": Conv(self.img_in_channels, 64, 3, 2),  # layer 0  (3, 640, 640) -> (64, 319, 319)
                "Conv_2": Conv(64, 128, 3, 2, 1, 2),  # layer 1 (64, 319, 319) -> (128, 160, 160)
                "DSC3k2_1": DSC3k2(128, 256, 2, True, 0.25),  # layer 2 (128, 160, 160) -> (256, 160, 160)
                "Conv_3": Conv(256, 256, 3, 2, 1, 4),  # layer 3 (256, 160, 160) -> (256, 80, 80)
                "DSC3k2_2": DSC3k2(256, 512, 2, True, 0.25),  # layer 4 b1-out (512, 80, 80)
                "DSConv_1": DSConv(512, 512, 3, 2),  # layer 5
                "A2C2f_1": A2C2f(512, 512, 4, True, 4, True, 1.5),  # layer 6 b2-out (512, 40, 40)
                "DSConv_2": DSConv(512, 512, 3, 2),  # layer 7
                "A2C2f_2": A2C2f(512, 512, 4, True, 1, True, 1.5),  # layer 8 b3-out (512, 20, 20)
            }
        )

        self.thermal_backbone = nn.ModuleDict(
            {
                "Conv_1": Conv(self.the_in_channels, 64, 3, 2),  # layer 0
                "Conv_2": Conv(64, 128, 3, 2, 1, 2),  # layer 1
                "DSC3k2_1": DSC3k2(128, 256, 1, True, 0.25),  # layer 2
                "Conv_3": Conv(256, 256, 3, 2, 1, 4),  # layer 3
                "DSC3k2_2": DSC3k2(256, 512, 1, True, 0.25),  # layer 4 b1-out (512, 80, 80)
                "DSConv_1": DSConv(512, 512, 3, 2),  # layer 5
                "A2C2f_1": A2C2f(512, 512, 2, True, 4, True, 1.5),  # layer 6 b2-out (512, 40, 40)
                "DSConv_2": DSConv(512, 512, 3, 2),  # layer 7
                "A2C2f_2": A2C2f(512, 512, 2, True, 1, True, 1.5),  # layer 8 b3-out (512, 20, 20)
            }
        )  # smaller size

        # neck
        self.neck = nn.ModuleDict(
            {
                # uable to train
                "HyperACE_Downsample": DownsampleConv(512, False),
                "HyperACE_Upsample": nn.Upsample(None, 2, "nearest"),
                "Upsample": nn.Upsample(None, 2, "nearest"),
                "Cat": Concat(1),
                # able to train
                "Inter_HyperACE": HyperACE(512, 512, 2, 8, True, True, 0.5, 1, "both", False),
                "Cross_HyperACE": CHyperACE(512, 512, 2, 8, True, True, 0.5, 1, "both"),
                "MMFullPAD_Tunnel_1": MMFullPAD_Tunnel(),
                "MMFullPAD_Tunnel_2": MMFullPAD_Tunnel(),
                "MMFullPAD_Tunnel_3": MMFullPAD_Tunnel(),
                "DSC3k2_1": DSC3k2(1024, 512, 2, True),
                "MMFullPAD_Tunnel_4": MMFullPAD_Tunnel(),
                "DSC3k2_2": DSC3k2(1024, 256, 2, True),
                "Conv_1": Conv(512, 256, 1, 1),
                "MMFullPAD_Tunnel_5": MMFullPAD_Tunnel(),
                "Conv_2": Conv(256, 256, 3, 2),
                "DSC3k2_3": DSC3k2(768, 512, 2, True),
                "MMFullPAD_Tunnel_6": MMFullPAD_Tunnel(),
                "Conv_3": Conv(512, 512, 3, 2),
                "DSC3k2_4": DSC3k2(1024, 512, 2, True),
                "MMFullPAD_Tunnel_7": MMFullPAD_Tunnel(),
            }
        )  # small size

        # head
        self.head = nn.ModuleDict({"head": DetectV8(nc=self.nc, ch=(256, 512, 512))})  # small size

    def forward(self, imgs: torch.Tensor, thermals: torch.Tensor) -> tuple[torch.Tensor]:
        # img backbone
        img_skips = []
        for key, layer in self.img_backbone.items():
            imgs = layer(imgs)
            if key in ["DSC3k2_2", "A2C2f_1", "A2C2f_2"]:
                img_skips.append(imgs)

        # themral backbone
        thermal_skips = []
        for key, layer in self.thermal_backbone.items():
            thermals = layer(thermals)
            if key in ["DSC3k2_2", "A2C2f_1", "A2C2f_2"]:
                thermal_skips.append(thermals)

        pixels_fuse = [img_skips[i] + thermal_skips[i] for i in range(len(img_skips))]

        # ----- neck -----
        # Inter HyperACE
        img_enhanced = []
        thermal_enhanced = []

        img_h2 = self.neck.Inter_HyperACE(img_skips)
        thermal_h2 = self.neck.Inter_HyperACE(thermal_skips)

        img_h1 = self.neck.HyperACE_Upsample(img_h2)
        thermal_h1 = self.neck.HyperACE_Upsample(thermal_h2)

        img_enhanced.append(img_h1)
        thermal_enhanced.append(thermal_h1)
        img_enhanced.append(img_h2)
        thermal_enhanced.append(thermal_h2)

        img_h3 = self.neck.HyperACE_Downsample(img_h2)
        thermal_h3 = self.neck.HyperACE_Downsample(thermal_h2)

        img_enhanced.append(img_h3)
        thermal_enhanced.append(thermal_h3)

        # Cross HyerACE
        fuse_enhanced = []
        fuse_h2 = self.neck.Cross_HyperACE([img_enhanced[1], thermal_enhanced[1]])
        fuse_h1 = self.neck.HyperACE_Upsample(fuse_h2)
        fuse_h3 = self.neck.HyperACE_Downsample(fuse_h2)
        fuse_enhanced.append(fuse_h1)
        fuse_enhanced.append(fuse_h2)
        fuse_enhanced.append(fuse_h3)

        f1 = self.neck.MMFullPAD_Tunnel_1([pixels_fuse[0], img_enhanced[0], thermal_enhanced[0], fuse_enhanced[0]])
        f2 = self.neck.MMFullPAD_Tunnel_2([pixels_fuse[1], img_enhanced[1], thermal_enhanced[1], fuse_enhanced[1]])
        f3 = self.neck.MMFullPAD_Tunnel_3([pixels_fuse[2], img_enhanced[2], thermal_enhanced[2], fuse_enhanced[2]])

        # Full_Tunnel
        d1 = self.neck.DSC3k2_1(self.neck.Cat([self.neck.Upsample(f3), f2]))
        f5 = self.neck.MMFullPAD_Tunnel_5([d1, img_enhanced[1], thermal_enhanced[1], fuse_enhanced[1]])
        d2 = self.neck.DSC3k2_2(self.neck.Cat([self.neck.Upsample(d1), f1]))
        # det1
        det1 = f4 = self.neck.MMFullPAD_Tunnel_4(
            [
                d2,
                self.neck.Conv_1(img_enhanced[0]),  # 是否分开Conv？
                self.neck.Conv_1(thermal_enhanced[0]),
                self.neck.Conv_1(fuse_enhanced[0]),
            ]
        )
        d3 = self.neck.DSC3k2_3(self.neck.Cat([self.neck.Conv_2(f4), f5]))
        # det2
        det2 = self.neck.MMFullPAD_Tunnel_6([d3, img_enhanced[1], thermal_enhanced[1], fuse_enhanced[1]])
        # det3
        d4 = self.neck.DSC3k2_4(self.neck.Cat([self.neck.Conv_3(d3), f3]))
        det3 = self.neck.MMFullPAD_Tunnel_7([d4, img_enhanced[2], thermal_enhanced[2], fuse_enhanced[2]])

        # ----- head -----

        return self.head([det1, det2, det3])

    def view_structure(self) -> None:
        from torchinfo import summary

        img_input = torch.randn(1, *self.img_shape, device=self.device)
        thermal_input = torch.randn(1, *self.thermal_shape, device=self.device)

        summary(self, input_data=[img_input, thermal_input])

    def _initialize_weights(self):
        super()._initialize_weights()
        self.heads.bias_init()
