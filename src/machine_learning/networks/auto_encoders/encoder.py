import torch.nn as nn
from machine_learning.models import BaseNet


class Encoder(BaseNet):
    def __init__(self, image_shape: tuple[int], z_dim: int) -> None:
        """
         Encoder network

        Args:
            image_shape (tuple[int]): the shape of input data (channels, height, width).
            z_dim (int): Output the dimension of the encoder output.
        """
        super().__init__()

        self.image_shape = image_shape
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

        summary(self, input_size=(1, *self.image_shape))
