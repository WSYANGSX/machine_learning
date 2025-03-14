from abc import ABC, abstractmethod
from collections import OrderedDict

import torch.nn as nn
from torchinfo import summary


class BaseNet(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def _initialize_weights(self):
        pass

    @abstractmethod
    def view_structure(self):
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

        prev_dim = input_dim
        for key, value in hidden_layers.items():
            if isinstance(value, nn.Linear):
                if value.in_features != prev_dim:
                    raise ValueError(f"层{key}输入维度不匹配.")
                prev_dim = value.out_features
        if prev_dim != output_dim:
            raise ValueError("最后一层输出维度与 output_dim 不匹配.")

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


class CNN(BaseNet):
    def __init__(
        self,
        input_size: tuple[int],
        output_size: tuple[int],
        hidden_layers: OrderedDict[nn.Module],
    ) -> None:
        """
        Network for image handling

        Args:
            input_size (tuple[int]): 输入数据的size (channels, ...).
            output_size (tuple[int]): 输出特征向量的size.
            hidden_layers (OrderedDict[nn.Module]): 隐藏层.
        """
        super().__init__()

        if not hidden_layers:
            raise ValueError("hidden_layers不能为空.")

        self.input_size = input_size
        self.output_size = output_size

        prev_size = self.input_size
        for key, value in hidden_layers.items():
            if isinstance(value, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                if value.in_features != prev_dim:
                    raise ValueError(f"层{key}输入维度不匹配.")
                prev_dim = value.out_features
        if prev_dim != self.output_size:
            raise ValueError("最后一层输出维度与 output_dim 不匹配.")

        self.front_net = nn.Sequential(hidden_layers)
        self.output_layer = nn.Linear()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        output1 = self.layer1(x)
        output2 = self.layer2(output1)

        mu = self.mu(output2)
        log_var = self.sigma(output2)

        return mu, log_var

    def __str__(self):
        summary(self, input_size=self.input_size)
