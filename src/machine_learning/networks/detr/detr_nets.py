class DETRTransformer(nn.Module):
    """DETR-style Transformer with only encoder and no positional embedding."""

    def __init__(
        self,
        num_layers: int,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.1,
        activation: str = "gelu",
        normalize_before: bool = False,
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            num_layers, dim, num_heads, mlp_ratio, qkv_bias, dropout, activation, normalize_before
        )

    def forward(self, src: torch.Tensor, mask: torch.Tensor | None = None, padding_mask: torch.Tensor | None = None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)

        enc_out = self.encoder(src, mask=mask, padding_mask=padding_mask)
        return enc_out.transpose(1, 2).view(bs, c, h, w)
