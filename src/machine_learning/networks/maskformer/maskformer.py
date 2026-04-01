from typing import Any

import torch
import torch.nn as nn

from machine_learning.networks import BaseNet


class TransformerPredictor(BaseNet):
    def __init__(
        self,
        *args: tuple[Any, ...],
        **kwargs: dict[str, Any],
    ):
        super().__init__(args=args, kwargs=kwargs)

    @property
    def dummy_input(self):
        return torch.randn(1, 3, 512, 512)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        pass

    def view_structure(self):
        from torchinfo import summary

        summary(self, (1, 3, 512, 512))
