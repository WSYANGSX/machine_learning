import torch.nn as nn
from machine_learning.networks import BaseNet


class AENet(BaseNet):
    def __init__(self, image_shape: tuple[int], z_dim: int) -> None:
        """
        auto_encoder network

        Args:
            image_shape (tuple[int]): the shape of the input image.
            z_dim (int): the dimension of the feature space.
        """
        super().__init__()

        self.image_shape = image_shape
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 3, 2, 2, 0),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Conv2d(3, 10, 2, 2, 0),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(10, 15, 2, 2, 0),
            nn.BatchNorm2d(15),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(135, self.z_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, 294),
            nn.Unflatten(1, (6, 7, 7)),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.ConvTranspose2d(6, 3, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.ConvTranspose2d(3, 1, 3, 2, 1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.encoder(x)
        y = self.decoder(y)

        return y

    def view_structure(self):
        from torchinfo import summary

        summary(self, input_size=(1, *self.image_shape))
