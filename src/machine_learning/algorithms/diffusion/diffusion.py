import torch  # noqa: F401
import torch.nn as nn


class Diffusion(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
