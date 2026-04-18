from typing import Literal, Any

import torch
import torch.nn as nn
import torch_dct as dct
import torch.nn.functional as F

from ultralytics.nn.modules import Conv
from machine_learning.networks import BaseNet
from machine_learning.modules.blocks import FrenquencyGuidedHyperFusion


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

        self.fusion = FrenquencyGuidedHyperFusion(
            in_channels, out_channels, e, num_hyperedges, rank, sparse_ratio, dropout, context, mode, node_topk_chunk
        )

    def forward(self, x):
        orig_dtype = x.dtype
        dct_x = dct.dct_2d(x.float(), norm="ortho")
        dct_x = dct_x.to(orig_dtype)
        x_out = self.fusion(x, dct_x)

        return x_out


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

    def forward(self, x, edge):
        x = self.cvt(x)
        x -= edge
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
        """Frenquency guided hypergraph fusion segmentation network."""
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

        # ------------------- 1. Encoder (Spatial Backbone) -------------------
        self.encoder_convs = nn.ModuleList(
            [
                nn.Sequential(Conv(self.channels, ch[0], 6, 2, 2), Conv(ch[0], ch[0])),  # 0: P1/2
                nn.Sequential(Conv(ch[0], ch[1], 3, 2), Conv(ch[1], ch[1])),  # 1: P2/4
                nn.Sequential(Conv(ch[1], ch[2], 3, 2), Conv(ch[2], ch[2])),  # 2: P3/8
                nn.Sequential(Conv(ch[2], ch[3], 3, 2), Conv(ch[3], ch[3])),  # 3: P4/16
                nn.Sequential(Conv(ch[3], ch[4], 3, 2), Conv(ch[4], ch[4])),  # 4: P5/32
            ]
        )

        # ------------------- 2. Fusion -------------------
        self.vqfghf_blocks = nn.ModuleList(
            [
                nn.Identity(),
                nn.Identity(),
                # P3 (1/8), P4 (1/16), P5 (1/32)
                FHGFBlock(ch[2], ch[2], num_hyperedges=12, rank=8),
                FHGFBlock(ch[3], ch[3], num_hyperedges=12, rank=8),
                FHGFBlock(ch[4], ch[4], num_hyperedges=12, rank=8),
            ]
        )

        self.edge_extractors = nn.ModuleList([nn.Conv2d(ch[i], 1, kernel_size=1) for i in range(5)])

        # ------------------- 3. Decoder -------------------
        self.decoder_blocks = nn.ModuleList(
            [
                UpBlock(in_channels=ch[4], skip_channels=ch[3], out_channels=ch[3]),  # P5 -> P4
                UpBlock(in_channels=ch[3], skip_channels=ch[2], out_channels=ch[2]),  # P4 -> P3
                UpBlock(in_channels=ch[2], skip_channels=ch[1], out_channels=ch[1]),  # P3 -> P2
                UpBlock(in_channels=ch[1], skip_channels=ch[0], out_channels=ch[0]),  # P2 -> P1
            ]
        )

        # edge upsample
        self.edge_upsample = nn.Upsample(size=(self.imgsz, self.imgsz), mode="bilinear", align_corners=True)

        # ------------------- 4. Prediction Heads -------------------
        self.mask_head = MaskHead(ch[0], self.nc)

        self.edge_head = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=16, kernel_size=1, bias=False),
            Conv(16, 16, 3, 1, 1),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    @property
    def dummy_input(self) -> tuple[torch.Tensor]:
        return (torch.randn(1, self.channels, self.imgsz, self.imgsz, device=self.device),)

    def forward(self, imgs: torch.Tensor):
        # 1. Encoder
        x = imgs
        spatial_skips = []
        for block in self.encoder_convs:
            x = block(x)
            spatial_skips.append(x)  # collect P1, P2, P3, P4, P5

        # 2. Frequency & Hypergraph Fusion
        fusions = []
        for i, feat in enumerate(spatial_skips):
            fusion = self.vqfghf_blocks[i](feat)
            fusions.append(fusion)

        # 3. Feat decoder
        out = fusions[4]
        out = self.decoder_blocks[0](out, fusions[3])  # P5 + F4 -> P4
        out = self.decoder_blocks[1](out, fusions[2])  # P4 + F3 -> P3
        out = self.decoder_blocks[2](out, fusions[1])  # P3 + F2 -> P2
        out = self.decoder_blocks[3](out, fusions[0])  # P2 + F1 -> P1

        # 4. Edge decoder
        edge_maps = [extractor(f) for extractor, f in zip(self.edge_extractors, fusions)]

        edge_maps_up = []
        for em in edge_maps:
            edge_maps_up.append(self.edge_upsample(em))
        fused_multi_edges = torch.cat(edge_maps_up, dim=1)

        # 5. Prediction heads
        edge_preds = self.edge_head(fused_multi_edges)  # [B, 1, H, W]
        mask_preds = self.mask_head(out, edge_preds)  # [B, nc, H, W]

        return mask_preds, edge_preds

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
