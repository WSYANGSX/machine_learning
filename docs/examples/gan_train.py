import torch.nn as nn
from torchvision import transforms
from torchinfo import summary

from machine_learning.algorithms import GAN
from machine_learning.trainer import Trainer
from machine_learning.utils import data_parse
from machine_learning.models import BaseNet


# 模型定义
class Generator(BaseNet):
    def __init__(self, input_dim: int, output_size: tuple[int]) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.model = nn.Sequential(layers)

    def forward(self, x):
        out_put = self.model(x)
        return out_put

    def view_structure(self):
        summary(self, input_size=(1, self.input_dim))


class Discriminator(nn.Module):
    def __init__(self, input_size: tuple[int], layers: OrderedDict[str, nn.Module]) -> None:
        super().__init__()

        self.input_size = input_size
        self.model = nn.Sequential(layers)

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
        output = self.model(x)

        return output

    def view_structure(self):
        summary(self, input_size=self.input_size)
