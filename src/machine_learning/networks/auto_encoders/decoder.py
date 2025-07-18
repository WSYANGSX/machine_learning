import torch.nn as nn
from machine_learning.networks import BaseNet


class Decoder(BaseNet):
    def __init__(self, z_dim: int) -> None:
        """
        Decoder network

        Args:
            z_dim (int): the dimension of the Decoder input.
        """
        super().__init__()

        self.z_dim = z_dim

        self.layer1 = nn.Sequential(nn.Linear(self.z_dim, 294), nn.Unflatten(1, (6, 7, 7)))
        self.layer2 = nn.Sequential(nn.BatchNorm2d(6), nn.ReLU(), nn.ConvTranspose2d(6, 3, 3, 2, 1, output_padding=1))
        self.layer3 = nn.Sequential(
            nn.BatchNorm2d(3), nn.ReLU(), nn.ConvTranspose2d(3, 1, 3, 2, 1, output_padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        mid_val = self.layer1(x)
        mid_val = self.layer2(mid_val)
        output = self.layer3(mid_val)

        return output

    def view_structure(self):
        from torchinfo import summary

        summary(self, input_size=(1, self.z_dim))
