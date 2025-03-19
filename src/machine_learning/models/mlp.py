from collections import OrderedDict

import torch.nn as nn
from torchinfo import summary

from .base import BaseNet


class MLP(BaseNet):
    def __init__(self, input_dim: int, output_dim: int, hidden_layers: OrderedDict[str, nn.Module]) -> None:
        """MLP network

        Args:
            input_size (int): 输入数据的维度.
            hidden_layers (OrderedDict[nn.Module]): 隐藏层.
            output_size (tuple[int]): 输出特征向量的维度.
            name (optional[str]): 模型的名称.
        """
        super().__init__()

        if not hidden_layers:
            raise ValueError("hidden_layers不能为空.")

        self.check_layers(input_dim, output_dim, hidden_layers)

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(hidden_layers)

    def _initialize_weights(self):
        print("Initializing weights with Kaiming normal...")
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)

    def view_structure(self):
        summary(self, input_size=(1, self.input_dim))

    def check_layers(self, input_dim: int, output_dim: int, hidden_layers: OrderedDict[str, nn.Module]) -> None:
        prev_dim = input_dim
        for key, value in hidden_layers.items():
            if isinstance(value, nn.Linear):
                if value.in_features != prev_dim:
                    raise ValueError(f"层{key}输入维度不匹配.")
                prev_dim = value.out_features
        if prev_dim != output_dim:
            raise ValueError("最后一层输出维度与 output_dim 不匹配.")
