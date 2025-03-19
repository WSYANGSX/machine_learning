import torch.nn as nn
from torchinfo import summary

from machine_learning.models import BaseNet


class Encoder(BaseNet):
    def __init__(
        self,
        input_size: tuple[int],
        z_dim: int,
    ) -> None:
        """
        Network for vae encoder.

        Args:
            input_size (tuple[int]): 输入数据的size (channels, ...).
            z_dim (int): 多维高斯的维度.
        """
        super().__init__()

        self.input_size = input_size
        self.z_dim = z_dim

        self.layer1 = nn.Sequential(nn.Conv2d(1, 3, 2, 2, 0), nn.BatchNorm2d(3), nn.ReLU())  # (3,14,14)
        self.layer2 = nn.Sequential(nn.Conv2d(3, 10, 2, 2, 0), nn.BatchNorm2d(10), nn.ReLU())  # (10,7,7)
        self.layer3 = nn.Sequential(nn.Conv2d(10, 15, 2, 2, 0), nn.BatchNorm2d(15), nn.ReLU(), nn.Flatten())  # (15,3,3)
        self.layer4 = nn.Linear(135, self.z_dim)

    def _initialize_weights(self):
        print("[INFO] Initializing weights with Kaiming normal...")
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        mid_val = self.layer1(x)
        mid_val = self.layer2(mid_val)
        mid_val = self.layer3(mid_val)
        output = self.layer4(mid_val)

        return output

    def view_structure(self):
        summary(self, input_size=(1, *self.input_size))
