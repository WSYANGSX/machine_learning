from abc import ABC, abstractmethod
from collections import OrderedDict

import torch.nn as nn
from torchinfo import summary

from machine_learning.utils import cal_conv_output_size, cal_pooling_output_size


class BaseNet(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def _initialize_weights(self):
        pass

    @abstractmethod
    def view_structure(self):
        pass

    @abstractmethod
    def check_layers(self):
        pass


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


class CNN(BaseNet):
    def __init__(
        self,
        input_size: tuple[int],
        output_size: tuple[int],
        hidden_layers: OrderedDict[nn.Module],
    ) -> None:
        """
        Network for CNN handling

        Args:
            input_size (tuple[int]): 输入数据的size (channels, ...).
            output_size (tuple[int]): 输出数据的size (channels, ...).
            hidden_layers (OrderedDict[nn.Module]): 隐藏层.
        """
        super().__init__()

        if not hidden_layers:
            raise ValueError("hidden_layers不能为空.")

        self.check_layers(input_size, output_size, hidden_layers)

        self.input_size = input_size
        self.output_size = output_size

        self.layers = nn.Sequential(hidden_layers)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.layers(x)

    def view_structure(self):
        summary(self, input_size=(1, *self.input_size))

    def check_layers(self, input_size, output_size, hidden_layers):
        prev_size = (1, *input_size)
        for key, value in hidden_layers.items():
            if isinstance(value, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                if prev_size[1] != value.in_channels:
                    raise ValueError(f"输入通道数与{key}层所需通道数不一致.")
                prev_size = cal_conv_output_size(prev_size, value)
            elif isinstance(
                value, (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d, nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)
            ):
                prev_size = cal_pooling_output_size(prev_size, value)
        if prev_size[1:] != output_size:
            raise ValueError("最后一层输出大小与 output_size 不匹配.")
