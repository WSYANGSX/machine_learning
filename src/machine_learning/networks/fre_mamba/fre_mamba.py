from typing import Literal, Any

import torch
import torch.nn as nn
import torch_dct as dct
import torch.nn.functional as F

from ultralytics.nn.modules import Conv
from machine_learning.networks import BaseNet
from machine_learning.modules.blocks import GaussianFrequencySplitter, FusionMamba, ModalFuseGate


class LightFMFBlock(nn.Module):
    def __init__(self, in_channels: int, dim: int):
        super().__init__()
        self.fre_splitter = GaussianFrequencySplitter()

        self.l_spa = Conv(in_channels, in_channels, 7, 1, 3)
        self.h_spa = Conv(in_channels, in_channels, 3, 1, 1)

        self.l_fusion = Conv(in_channels * 2, dim, 1, 1)
        self.h_fusion = Conv(in_channels * 2, dim, 1, 1)

        self.fuse = ModalFuseGate(dim)
        self.cv = Conv(in_channels, in_channels, 1, 1)

    def forward(self, x: torch.Tensor):
        orig_dtype = x.dtype
        fre = dct.dct_2d(x.float(), norm="ortho")

        l_fre, h_fre = self.fre_splitter(fre)

        l_fre_spa = dct.idct_2d(l_fre, norm="ortho").to(orig_dtype)
        h_fre_spa = dct.idct_2d(h_fre, norm="ortho").to(orig_dtype)

        l_spa = self.l_spa(x)
        h_spa = self.h_spa(x)

        l_fusion = self.l_fusion(torch.cat([l_spa, l_fre_spa], dim=1))
        h_fusion = self.h_fusion(torch.cat([h_spa, h_fre_spa], dim=1))

        y = self.fuse((l_fusion, h_fusion))
        return self.cv(y)


class FMFBlock(nn.Module):
    def __init__(self, in_channels: int, dim: int, H: int, W: int):
        super().__init__()
        self.H = H
        self.W = W

        self.fre_splitter = GaussianFrequencySplitter()
        self.l_fusion = FusionMamba(in_channels, dim, H, W)
        self.h_fusion = FusionMamba(in_channels, dim, H, W)

        self.l_spa = Conv(in_channels, in_channels, 7, 1, 3)
        self.h_spa = Conv(in_channels, in_channels, 3, 1, 1)

        self.fuse = ModalFuseGate(dim)
        self.cv = Conv(in_channels, in_channels, 1, 1)

    def forward(self, x: torch.Tensor):
        orig_dtype = x.dtype
        fre = dct.dct_2d(x.float(), norm="ortho")
        fre = fre.to(orig_dtype)

        # fre split
        l_fre, h_fre = self.fre_splitter(fre)
        # spa split
        l_spa = self.l_spa(x)
        h_spa = self.h_spa(x)

        # mamba fusion
        l_fusion = self.l_fusion([l_spa, l_fre])
        h_fusion = self.h_fusion([h_spa, h_fre])

        y = self.fuse((l_fusion, h_fusion))
        return self.cv(y)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            Conv(in_channels + skip_channels, out_channels, 3, 1, 1), Conv(out_channels, out_channels, 3, 1, 1)
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class MaskHead(nn.Module):
    def __init__(self, channel: int, nc: int):
        super().__init__()
        self.cvt = nn.Sequential(
            nn.ConvTranspose2d(channel, channel // 2, kernel_size=2, stride=2),
            Conv(channel // 2, channel // 2, 3, 1, 1),
            nn.ConvTranspose2d(channel // 2, channel // 4, kernel_size=2, stride=2),
        )
        self.cv1 = Conv(channel // 4, channel // 4, 3, 1, 1)
        self.cv2 = nn.Conv2d(channel // 4, nc, kernel_size=1)

    def forward(self, x, edge):
        x = self.cvt(x)
        x = x - edge
        x = self.cv1(x)
        x = self.cv2(x)
        return x


class FMFNet(BaseNet):
    def __init__(
        self,
        imgsz: int,
        nc: int,
        channels: int = 3,
        net_scale: Literal["n", "s", "l", "x"] = "n",
        single_cls: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(args=args, kwargs=kwargs)

        self.imgsz = imgsz
        self.nc = nc if not single_cls else 2
        self.net_scale = net_scale
        self.channels = channels

        # channels
        if self.net_scale == "n":
            ch = [16, 32, 64, 128, 256]
        elif self.net_scale == "s":
            ch = [32, 64, 128, 256, 512]
        elif self.net_scale == "l":
            ch = [48, 96, 192, 384, 768]
        else:
            raise ValueError(f"Unsupported scale: {net_scale}")

        # 1. Encoder
        self.encoder_convs = nn.ModuleList(
            [
                nn.Sequential(Conv(self.channels, ch[0], 6, 2, 2), Conv(ch[0], ch[0])),  # P1
                nn.Sequential(Conv(ch[0], ch[1], 3, 2), Conv(ch[1], ch[1])),  # P2
                nn.Sequential(Conv(ch[1], ch[2], 3, 2), Conv(ch[2], ch[2])),  # P3
                nn.Sequential(Conv(ch[2], ch[3], 3, 2), Conv(ch[3], ch[3])),  # P4
                nn.Sequential(Conv(ch[3], ch[4], 3, 2), Conv(ch[4], ch[4])),  # P5
            ]
        )

        # 2. Fusion
        self.fusion_blocks = nn.ModuleList()
        for i in range(1, 5):
            h, w = self.imgsz // (2 ** (i + 1)), self.imgsz // (2 ** (i + 1))
            if i == 1:
                self.fusion_blocks.append(LightFMFBlock(ch[i], ch[i]))
            else:
                self.fusion_blocks.append(FMFBlock(ch[i], ch[i], h, w))
        self.edge_extractors = nn.ModuleList([nn.Conv2d(ch[i], 1, kernel_size=1) for i in range(1, 5)])

        # 3. Decoder
        self.decoder_blocks = nn.ModuleList(
            [
                UpBlock(in_channels=ch[4], skip_channels=ch[3], out_channels=ch[3]),  # Stage 1
                UpBlock(in_channels=ch[3], skip_channels=ch[2], out_channels=ch[2]),  # Stage 2
                UpBlock(in_channels=ch[2], skip_channels=ch[1], out_channels=ch[1]),  # Stage 3
            ]
        )

        self.edge_upsample = nn.Upsample(size=(self.imgsz, self.imgsz), mode="bilinear", align_corners=True)

        # 4. Prediction Heads
        self.mask_head = MaskHead(ch[1], self.nc)
        self.edge_head = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=1, bias=False),
            Conv(16, 16, 3, 1, 1),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, imgs: torch.Tensor):
        # 1. Encoder
        x = imgs
        spatial_skips = []
        for block in self.encoder_convs:
            x = block(x)
            spatial_skips.append(x)

        # 2. Frequency & Mamba Fusion
        fusions = [None]
        for i, feat in enumerate(spatial_skips):
            if i == 0:
                continue
            fusions.append(self.fusion_blocks[i - 1](feat))

        # 3. Feat decoder
        out = fusions[4]
        out = self.decoder_blocks[0](out, fusions[3])
        out = self.decoder_blocks[1](out, fusions[2])
        out = self.decoder_blocks[2](out, fusions[1])  # 最终输出尺度为 P2

        # 4. Edge decoder (跳过 P1)
        edge_maps_up = []
        for i, extractor in enumerate(self.edge_extractors):
            em = extractor(fusions[i + 1])
            edge_maps_up.append(self.edge_upsample(em))

        fused_multi_edges = torch.cat(edge_maps_up, dim=1)

        # 5. Prediction heads
        edge_preds = self.edge_head(fused_multi_edges)
        mask_preds = self.mask_head(out, edge_preds)

        return mask_preds, edge_preds

    @property
    def dummy_input(self) -> tuple[torch.Tensor]:
        return torch.randn(1, self.channels, self.imgsz, self.imgsz, device=self.device)

    def _initialize_weights(self):
        for m in self.modules():
            t = type(m)
            if t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in {nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU}:
                m.inplace = True


if __name__ == "__main__":
    fghf = FMFNet(imgsz=640, nc=3, channels=3, net_scale="s").to("cuda")
    fghf.view_structure()
    x = torch.randn(2, 3, 640, 640).to("cuda")
    mask, edge = fghf(x)
    print(f"Mask shape: {mask.shape}, Edge shape: {edge.shape}")
