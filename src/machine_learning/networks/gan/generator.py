import numpy as np
import torch.nn as nn

from machine_learning.models import BaseNet


class Generator(BaseNet):
    def __init__(self, input_dim: int, output_size: tuple[int]) -> None:
        """
        GAN generator network.

        Args:
            input_dim (int): the dimension of the input feature vector.
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_size = output_size

        self.layer1 = nn.Sequential(nn.Linear(self.input_dim, 256), nn.LeakyReLU(0.2))
        self.layer2 = nn.Sequential(nn.Linear(256, 512), nn.LeakyReLU(0.2))
        self.layer3 = nn.Sequential(nn.Linear(512, 1024), nn.LeakyReLU(0.2))
        self.layer4 = nn.Sequential(
            nn.Linear(1024, np.prod(self.output_size)), nn.Tanh(), nn.Unflatten(1, self.output_size)
        )

    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y)
        y = self.layer3(y)
        out = self.layer4(y)

        return out

    def view_structure(self):
        from torchinfo import summary

        summary(self, input_size=(1, self.input_dim))
