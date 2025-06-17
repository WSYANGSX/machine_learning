import torch.nn as nn
from machine_learning.models import BaseNet


class Encoder(BaseNet):
    def __init__(self, input_size: tuple[int], z_dim: int) -> None:
        """
         Encoder network

        Args:
            input_size (tuple[int]): the size of input data (channels, height, width).
            z_dim (int): Output the dimension of the encoder output.
        """
        super().__init__()

        self.input_size = input_size
        self.z_dim = z_dim

        self.layer1 = nn.Sequential(nn.Conv2d(1, 3, 2, 2, 0), nn.BatchNorm2d(3), nn.ReLU())  # (3,14,14)
        self.layer2 = nn.Sequential(nn.Conv2d(3, 10, 2, 2, 0), nn.BatchNorm2d(10), nn.ReLU())  # (10,7,7)
        self.layer3 = nn.Sequential(nn.Conv2d(10, 15, 2, 2, 0), nn.BatchNorm2d(15), nn.ReLU(), nn.Flatten())  # (15,3,3)
        self.layer4 = nn.Linear(135, self.z_dim)

    def forward(self, x):
        mid_val = self.layer1(x)
        mid_val = self.layer2(mid_val)
        mid_val = self.layer3(mid_val)
        output = self.layer4(mid_val)

        return output

    def view_structure(self):
        from torchinfo import summary

        summary(self, input_size=(1, *self.input_size))


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
