from typing import Sequence

import torch
import torch.nn as nn

from machine_learning.networks import BaseNet
from machine_learning.modules.head import DetectV8

from ultralytics.nn.modules import Conv, DSC3k2, DSConv, A2C2f, HyperACE, DownsampleConv, Concat, FullPAD_Tunnel


class V13Net(BaseNet):
    def __init__(
        self,
        img_shape: Sequence[int],
        nc: int = 1,
    ):
        """multimodal object detection network.

        Args:
            img_shape (Sequence[int]): the shape of input rgb image.
            thermal_shape (Sequence[int]): the shape of input thermal image.
            num_classes (int): number of classes.
        """
        super().__init__()
        self.img_shape = img_shape  # (3, height, width)
        self.nc = nc

        self.img_in_channels = self.img_shape[0]

        # backbones
        self.backbone = nn.ModuleDict(
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

        # neck
        self.neck = nn.ModuleDict(
            {
                "HyperACE": HyperACE(512, 512, 2, 6, True, True, 0.5, 1, "both", False),
                "Downsample": DownsampleConv(512, False),
                "Upsample_1": nn.Upsample(None, 2, "nearest"),
                "FullPAD_Tunnel_1": FullPAD_Tunnel(),
                "FullPAD_Tunnel_2": FullPAD_Tunnel(),
                "FullPAD_Tunnel_3": FullPAD_Tunnel(),
                "Upsample_2": nn.Upsample(None, 2, "nearest"),
                "Cat_1": Concat(1),
                "DSC3k2_1": DSC3k2(1024, 512, 2, True),
                "FullPAD_Tunnel_4": FullPAD_Tunnel(),
                "Upsample_3": nn.Upsample(None, 2, "nearest"),
                "Cat_2": Concat(1),
                "DSC3k2_2": DSC3k2(1024, 256, 2, True),
                "Conv_1": Conv(512, 256, 1, 1),
                "FullPAD_Tunnel_5": FullPAD_Tunnel(),
                "Conv_2": Conv(256, 256, 3, 2),
                "Cat_3": Concat(1),
                "DSC3k2_3": DSC3k2(768, 512, 2, True),
                "FullPAD_Tunnel_6": FullPAD_Tunnel(),
                "Conv_3": Conv(512, 512, 3, 2),
                "Cat_4": Concat(1),
                "DSC3k2_4": DSC3k2(1024, 512, 2, True),
                "FullPAD_Tunnel_7": FullPAD_Tunnel(),
            }
        )  # x size

        # head
        self.head = nn.ModuleDict({"Head": DetectV8(nc=self.nc, ch=(256, 512, 512))})

    def forward(self, imgs: torch.Tensor) -> tuple[torch.Tensor]:
        # img backbone
        img_skips = []
        for key, layer in self.backbone.items():
            imgs = layer(imgs)
            if key in ["DSC3k2_2", "A2C2f_1", "A2C2f_2"]:
                img_skips.append(imgs)

        # ----- neck -----
        # HyperACE
        img_enhanced = []

        img_h2 = self.neck.HyperACE(img_skips)
        img_h1 = self.neck.Upsample_1(img_h2)
        img_h3 = self.neck.Downsample(img_h2)

        img_enhanced.append(img_h1)
        img_enhanced.append(img_h2)
        img_enhanced.append(img_h3)

        f1 = self.neck.FullPAD_Tunnel_1([img_skips[0], img_enhanced[0]])
        f2 = self.neck.FullPAD_Tunnel_2([img_skips[1], img_enhanced[1]])
        f3 = self.neck.FullPAD_Tunnel_3([img_skips[2], img_enhanced[2]])

        # Full_Tunnel
        d1 = self.neck.DSC3k2_1(self.neck.Cat_1([self.neck.Upsample_2(f3), f2]))
        f5 = self.neck.FullPAD_Tunnel_5([d1, img_enhanced[1]])
        d2 = self.neck.DSC3k2_2(self.neck.Cat_2([self.neck.Upsample_3(d1), f1]))
        # det1
        det1 = f4 = self.neck.FullPAD_Tunnel_4([d2, self.neck.Conv_1(img_enhanced[0])])
        d3 = self.neck.DSC3k2_3(self.neck.Cat_3([self.neck.Conv_2(f4), f5]))
        # det2
        det2 = self.neck.FullPAD_Tunnel_6([d3, img_enhanced[1]])
        # det3
        d4 = self.neck.DSC3k2_4(self.neck.Cat_4([self.neck.Conv_3(d3), f3]))
        det3 = self.neck.FullPAD_Tunnel_7([d4, img_enhanced[2]])

        # ----- head -----

        return self.head.Head([det1, det2, det3])

    def view_structure(self) -> None:
        super().view_structure()

        from torchinfo import summary

        img_input = torch.randn(1, *self.img_shape, device=self.device)

        summary(self, input_data=img_input)

    def _initialize_weights(self):
        super()._initialize_weights()
        self.head.Head.bias_init()
