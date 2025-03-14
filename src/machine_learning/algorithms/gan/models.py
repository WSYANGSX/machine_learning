from collections import OrderedDict
import torch.nn as nn

from torchinfo import summary


class Generator(nn.Module):
    def __init__(self, input_dim: int, layers: OrderedDict[str, nn.Module]) -> None:
        super().__init__()

        self.input_dim = input_dim
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
