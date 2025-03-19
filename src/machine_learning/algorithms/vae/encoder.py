from collections import OrderedDict

import torch.nn as nn
from torchinfo import summary


class Encoder(nn.Module):
    def __init__(self, input_size: tuple[int], z_dim: int) -> None:
        super().__init__()

        self.input_size = input_size
        self.z_dim = z_dim

        self.layer1 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=2, padding=1),
                    ),
                    (
                        "BatchNorm1",
                        nn.BatchNorm2d(3),
                    ),
                    ("relu1", nn.ReLU()),  # output image size N*3*14*14
                ]
            )
        )
        self.layer2 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv2",
                        nn.Conv2d(3, 6, 3, 2, 1),
                    ),
                    (
                        "BatchNorm2",
                        nn.BatchNorm2d(6),
                    ),
                    ("relu2", nn.ReLU()),  # output image size N*6*7*7
                    ("reshape", nn.Flatten()),  # output image size N*294
                ]
            )
        )

        self.mu = nn.Linear(294, self.z_dim)
        self.sigma = nn.Linear(294, self.z_dim)

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

    def view_structure(self):
        summary(self, input_size=self.input_size)

    def view_modules(self):
        for module in self.named_children():
            print(module)
