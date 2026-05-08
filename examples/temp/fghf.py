from typing import Literal, Any

import torch
import torch.nn as nn
import torch_dct as dct
import torch.nn.functional as F

from ultralytics.nn.modules import Conv
from machine_learning.networks import BaseNet
from machine_learning.networks.unet.unet import DownBlock
from machine_learning.modules.blocks import GaussianFrequencySplitter, FrenquencyGuidedHyperFusion


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


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            Conv(out_channels + skip_channels, out_channels, 3, 1, 1),
            Conv(out_channels, out_channels, 3, 1, 1),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class FGHFEMBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        e=1.0,
        num_hyperedges=32,
        rank=16,
        sparse_ratio=0.5,
        dropout=0.1,
        context="both",
        node_topk_chunk=None,
    ):
        super().__init__()

        self.fre_splitter = GaussianFrequencySplitter()

        # low frequency: global semantic consistency
        self.lfusion = FrenquencyGuidedHyperFusion(
            in_channels,
            out_channels,
            e,
            num_hyperedges,
            rank,
            sparse_ratio,
            dropout,
            context,
            mode="global",
            node_topk_chunk=node_topk_chunk,
        )

        # high frequency: local boundary/detail adaptation
        self.hfusion = FrenquencyGuidedHyperFusion(
            in_channels,
            out_channels,
            e,
            num_hyperedges,
            rank,
            sparse_ratio,
            dropout,
            context,
            mode="node",
            node_topk_chunk=node_topk_chunk,
        )

        hidden = max(out_channels // 4, 4)
        self.freq_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels * 2, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 2, 1),
            nn.Softmax(dim=1),
        )
        nn.init.zeros_(self.freq_weight[3].weight)
        nn.init.zeros_(self.freq_weight[3].bias)

        self.attn_conv = nn.Sequential(
            Conv(out_channels, out_channels, 3, 1, 1), nn.Conv2d(out_channels, out_channels, 1), nn.Sigmoid()
        )

        self.proj = Conv(out_channels, out_channels, 1, 1)
        self.res_proj = nn.Identity() if in_channels == out_channels else Conv(in_channels, out_channels, 1, 1)
        self.gamma = nn.Parameter(torch.tensor(0.05))

        # edge head for auxiliary supervision
        self.guided_filter = GuidedFilterFusion(out_channels)
        self.edge_head = nn.Sequential(
            Conv(out_channels, out_channels, 3, 1, 1), nn.Conv2d(out_channels, 1, kernel_size=1)
        )

    def spatial_high_pass(self, x, kernel_size=3):
        low = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        return x - low

    def forward(self, spa_feature, fre_feature, return_edge=False):
        l_fre, h_fre = self.fre_splitter(fre_feature)

        # use frequency energy instead of signed DCT coefficients
        l_fre = torch.log1p(torch.abs(l_fre))
        h_fre = torch.log1p(torch.abs(h_fre))

        lf_hg = self.lfusion(spa_feature, l_fre)
        hf_hg = self.hfusion(spa_feature, h_fre)

        # adaptive low/high fusion
        weight = self.freq_weight(torch.cat([lf_hg, hf_hg], dim=1))
        wl = weight[:, 0:1]
        wh = weight[:, 1:2]

        hg = wl * lf_hg + wh * hf_hg

        # explicit attention guidance
        attn = self.attn_conv(hg)
        enhance = self.proj(attn * hg)

        if return_edge:
            h_spa_raw = self.spatial_high_pass(self.res_proj(spa_feature))
            hf_refined = self.guided_filter(h_spa_raw, hf_hg)
            edge_logit = self.edge_head(hf_refined)
        else:
            edge_logit = None

        out = self.res_proj(spa_feature) + self.gamma * enhance

        return out, edge_logit


class MaskHead(nn.Module):
    def __init__(self, channel: int, nc: int):
        super().__init__()
        self.cv1 = Conv(channel, channel, 3, 1, 1)
        self.cv2 = nn.Conv2d(channel, nc, kernel_size=1)

    def forward(self, x):
        return self.cv2(self.cv1(x))


class FGHFNet(BaseNet):
    def __init__(
        self,
        imgsz: int,
        nc: int,
        channels: int = 3,
        net_scale: Literal["n", "s", "l", "x"] = "n",
        activation: Literal["relu", "gelu", "leaky_relu"] = "relu",
        single_cls: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(args=args, kwargs=kwargs)

        self.imgsz = imgsz
        self.nc = nc if not single_cls else 2
        self.net_scale = net_scale
        self.channels = channels
        self.activation = activation

        if self.net_scale == "n":
            ch = [16, 32, 64, 128, 256]
        elif self.net_scale == "s":
            ch = [32, 64, 128, 256, 512]
        elif self.net_scale == "l":
            ch = [48, 96, 192, 384, 768]
        elif self.net_scale == "x":
            ch = [64, 128, 256, 512, 1024]
        else:
            raise ValueError(f"Unsupported scale: {net_scale}")

        # ------------------- 1. Spatial Encoder-------------------
        self.spa_encoders = nn.ModuleList(
            [
                DownBlock(self.channels, ch[0], activation=activation),  # 0: P1/2
                DownBlock(ch[0], ch[1], activation=activation),  # 1: P2/4
                DownBlock(ch[1], ch[2], activation=activation),  # 2: P3/8
                DownBlock(ch[2], ch[3], activation=activation),  # 3: P4/16
                DownBlock(ch[3], ch[4], activation=activation),  # 4: P5/32
            ]
        )

        # ------------------- 2. Fusion -------------------
        self.fusion_blocks = nn.ModuleList(
            [
                # FGHFEMBlock(ch[0], ch[0], num_hyperedges=4, rank=2),
                # FGHFEMBlock(ch[1], ch[1], num_hyperedges=6, rank=3),
                nn.Identity(),
                nn.Identity(),
                FGHFEMBlock(ch[2], ch[2], num_hyperedges=8, rank=4),
                FGHFEMBlock(ch[3], ch[3], num_hyperedges=12, rank=6),
                FGHFEMBlock(ch[4], ch[4], num_hyperedges=12, rank=6),
            ]
        )

        # ------------------- 3. Decoder -------------------
        self.decoders = nn.ModuleList(
            [
                UpBlock(in_channels=ch[4], skip_channels=ch[3], out_channels=ch[3]),  # P5 -> P4
                UpBlock(in_channels=ch[3], skip_channels=ch[2], out_channels=ch[2]),  # P4 -> P3
                UpBlock(in_channels=ch[2], skip_channels=ch[1], out_channels=ch[1]),  # P3 -> P2
                UpBlock(in_channels=ch[1], skip_channels=ch[0], out_channels=ch[0]),  # P2 -> P1
            ]
        )

        # ------------------- 4. Prediction Heads -------------------
        self.mask_head = MaskHead(ch[0], self.nc)

    def forward(self, imgs: torch.Tensor):
        x = imgs
        spa_skips = []
        for i, block in enumerate(self.spa_encoders):
            x = block(x)
            spa_skips.append(x)

        # 2. Dynamic Frequency & Hypergraph Fusion
        fusions, edges = [], []
        for i, cur_spa in enumerate(spa_skips):
            if i < 2:
                fusions.append(cur_spa)
                edges.append(None)
                continue

            cur_fre = dct.dct_2d(cur_spa.float(), norm="ortho").to(cur_spa.dtype)
            f, edge = self.fusion_blocks[i](cur_spa, cur_fre)
            fusions.append(f)
            edges.append(edge)

        # 3. Decoder
        out = fusions[4]
        out = self.decoders[0](out, fusions[3])
        out = self.decoders[1](out, fusions[2])
        out = self.decoders[2](out, fusions[1])
        out = self.decoders[3](out, fusions[0])

        # 4. Prediction heads
        mask_preds = self.mask_head(out)  # [B, nc, H, W]

        return mask_preds

    @property
    def dummy_input(self) -> tuple[torch.Tensor]:
        return torch.randn(1, self.channels, self.imgsz, self.imgsz, device=self.device)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    fghf = FGHFNet(imgsz=640, nc=3, channels=3, net_scale="s")
    fghf.view_structure()
    x = torch.randn(2, 3, 640, 640)
    mask = fghf(x)
    print(f"Mask shape: {mask.shape}")
