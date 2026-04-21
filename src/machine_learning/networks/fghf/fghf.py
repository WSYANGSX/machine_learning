from typing import Literal, Any

import torch
import torch.nn as nn
import torch_dct as dct
import torch.nn.functional as F

from ultralytics.nn.modules import Conv
from machine_learning.networks import BaseNet
from machine_learning.modules.blocks import GaussianFrequencySplitter, FrenquencyGuidedHyperFusion, ModalFuseGate


class GuidedFilterFusion(nn.Module):
    """
    Uses the raw, sharp spatial high-frequency features to guide and re-sharpen
    the over-smoothed hypergraph high-frequency outputs.
    """

    def __init__(self, channels):
        super().__init__()
        self.coeff_net = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size=1),
            nn.BatchNorm2d(channels * 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels * 2, channels * 2, kernel_size=1),
        )
        self.out_conv = Conv(channels, channels, 3, 1, 1)

    def forward(self, h_spa_raw: torch.Tensor, h_hg: torch.Tensor) -> torch.Tensor:
        """
        h_spa_raw: The raw, sharp spatial high-frequency (Guidance)
        h_hg: The globally correlated but smoothed hypergraph high-frequency (Input)
        """
        cat_feat = torch.cat([h_spa_raw, h_hg], dim=1)
        coeffs = self.coeff_net(cat_feat)
        a, b = torch.chunk(coeffs, 2, dim=1)
        a = torch.tanh(a)
        out = a * h_spa_raw + b + h_hg

        return self.out_conv(out)


class FHGFBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        e=1.0,
        num_hyperedges=32,
        rank: int = 16,
        sparse_ratio: float = 0.5,
        dropout: float = 0.1,
        context: str = "both",
        mode: Literal["global", "node"] = "global",
        node_topk_chunk: int | None = None,
    ):
        super().__init__()

        self.fre_splitter = GaussianFrequencySplitter()
        self.l_spa = Conv(in_channels, in_channels, 7, 1, 3)
        self.h_spa = Conv(in_channels, in_channels, 3, 1, 1)

        self.lfusion = FrenquencyGuidedHyperFusion(
            in_channels, out_channels, e, num_hyperedges, rank, sparse_ratio, dropout, context, mode, node_topk_chunk
        )
        self.hfusion = FrenquencyGuidedHyperFusion(
            in_channels, out_channels, e, num_hyperedges, rank, sparse_ratio, dropout, context, mode, node_topk_chunk
        )

        # self.hg_gff = GuidedFilterFusion(in_channels)

        self.fuse = ModalFuseGate(in_channels)
        self.cv = Conv(in_channels, in_channels, 1, 1)

    def forward(self, spa_feature: torch.Tensor, fre_feature: torch.Tensor):
        l_fre, h_fre = self.fre_splitter(fre_feature)
        l_spa, h_spa = self.l_spa(spa_feature), self.h_spa(spa_feature)

        lf_hg = self.lfusion(l_spa, l_fre)
        hf_hg = self.hfusion(h_spa, h_fre)

        # hf_reshaped = self.hg_gff(h_spa_raw=h_spa, h_hg=hf_hg)
        # f = self.fuse((lf_hg, hf_reshaped))

        f = self.fuse((lf_hg, hf_hg))

        return self.cv(f)


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

        self.cvt = nn.ConvTranspose2d(channel, channel, kernel_size=2, stride=2)
        self.cv1 = Conv(channel, channel, 3, 1, 1)
        self.cv2 = nn.Conv2d(channel, nc, kernel_size=1)

    def forward(self, x):
        x = self.cvt(x)
        x = self.cv1(x)
        x = self.cv2(x)
        return x


class FGHFNet(BaseNet):
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

        if self.net_scale == "n":
            ch = [16, 32, 64, 128, 256]
        elif self.net_scale == "s":
            ch = [32, 64, 128, 256, 512]
        elif self.net_scale == "l":
            ch = [48, 96, 192, 384, 768]
        else:
            raise ValueError(f"Unsupported scale: {net_scale}")

        # ------------------- 1. Spatial Encoder-------------------
        self.spa_encoders = nn.ModuleList(
            [
                nn.Sequential(Conv(self.channels, ch[0], 6, 2, 2), Conv(ch[0], ch[0])),
                nn.Sequential(Conv(ch[0], ch[1], 3, 2), Conv(ch[1], ch[1])),  # 1: P2/4
                nn.Sequential(Conv(ch[1], ch[2], 3, 2), Conv(ch[2], ch[2])),  # 2: P3/8
                nn.Sequential(Conv(ch[2], ch[3], 3, 2), Conv(ch[3], ch[3])),  # 3: P4/16
                nn.Sequential(Conv(ch[3], ch[4], 3, 2), Conv(ch[4], ch[4])),  # 4: P5/32
            ]
        )

        # ------------------- 2. Fusion -------------------
        self.fusion_blocks = nn.ModuleList(
            [
                FHGFBlock(ch[1], ch[1], num_hyperedges=6, rank=3),
                FHGFBlock(ch[2], ch[2], num_hyperedges=8, rank=4),
                FHGFBlock(ch[3], ch[3], num_hyperedges=12, rank=6),
                FHGFBlock(ch[4], ch[4], num_hyperedges=12, rank=6),
            ]
        )

        # ------------------- 3. Decoder -------------------
        self.decoders = nn.ModuleList(
            [
                UpBlock(in_channels=ch[4], skip_channels=ch[3], out_channels=ch[3]),  # P5 -> P4
                UpBlock(in_channels=ch[3], skip_channels=ch[2], out_channels=ch[2]),  # P4 -> P3
                UpBlock(in_channels=ch[2], skip_channels=ch[1], out_channels=ch[1]),  # P3 -> P2
            ]
        )

        # ------------------- 4. Prediction Heads -------------------
        self.mask_head = MaskHead(ch[1], self.nc)

    def forward(self, imgs: torch.Tensor):
        x = imgs
        spa_skips = []
        for i, block in enumerate(self.spa_encoders):
            x = block(x)
            if i > 0:
                spa_skips.append(x)  # collect P2, P3, P4, P5

        # 2. Dynamic Frequency & Hypergraph Fusion
        fusions = []
        for i, block in enumerate(self.fusion_blocks):
            cur_spa = spa_skips[i]
            cur_fre = dct.dct_2d(cur_spa.float(), norm="ortho").to(cur_spa.dtype)

            f = block(cur_spa, cur_fre)
            fusions.append(f)

        # 3. Decoder
        out = fusions[3]
        out = self.decoders[0](out, fusions[2])
        out = self.decoders[1](out, fusions[1])
        out = self.decoders[2](out, fusions[0])

        # 4. Prediction heads
        mask_preds = self.mask_head(out)  # [B, nc, H, W]

        return mask_preds

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
    fghf = FGHFNet(imgsz=640, nc=3, channels=3, net_scale="s")
    fghf.view_structure()
    x = torch.randn(2, 3, 640, 640)
    mask, edge = fghf(x)
    print(f"Mask shape: {mask.shape}, Edge shape: {edge.shape}")
