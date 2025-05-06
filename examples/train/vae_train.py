import torch.nn as nn
from torchinfo import summary
from torchvision import transforms

from machine_learning.models import BaseNet
from machine_learning.algorithms import VAE
from machine_learning.trainer import Trainer
from machine_learning.utils import data_parse


# 模型定义
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

        self.layer1 = nn.Sequential(nn.Conv2d(1, 3, 3, 2, 1), nn.BatchNorm2d(3), nn.ReLU())  # (3,14,14)
        self.layer2 = nn.Sequential(nn.Conv2d(3, 6, 3, 2, 1), nn.BatchNorm2d(6), nn.ReLU(), nn.Flatten())  # (6,7,7)

        self.mu = nn.Linear(294, self.z_dim)
        self.sigma = nn.Linear(294, self.z_dim)

    def forward(self, x):
        mid_val = self.layer1(x)
        mid_val = self.layer2(mid_val)

        mu = self.mu(mid_val)
        log_var = self.sigma(mid_val)

        return mu, log_var

    def view_structure(self):
        summary(self, input_size=(1, *self.input_size))


class Decoder(BaseNet):
    def __init__(self, z_dim: int) -> None:
        """
        Network for vae decoder.

        Args:
            z_dim (int): 特征向量的维度.
        """
        super().__init__()

        self.z_dim = z_dim

        self.layer1 = nn.Sequential(nn.Linear(self.z_dim, 294), nn.Unflatten(1, (6, 7, 7)))

        self.layer2 = nn.Sequential(
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.ConvTranspose2d(6, 3, 3, 2, 1, output_padding=1),
        )

        self.layer3 = nn.Sequential(
            nn.BatchNorm2d(3), nn.ReLU(), nn.ConvTranspose2d(3, 1, 3, 2, 1, output_padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        mid_val = self.layer1(x)
        mid_val = self.layer2(mid_val)
        output = self.layer3(mid_val)

        return output

    def view_structure(self):
        summary(self, input_size=(1, self.z_dim))


def main():
    image_size = (1, 28, 28)
    z_dim = 64

    encoder = Encoder(image_size, z_dim)
    decoder = Decoder(z_dim)
    models = {"encoder": encoder, "decoder": decoder}

    vae = VAE(
        "./src/machine_learning/algorithms/vae/config/config.yaml",
        models,
    )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=0.1307, std=0.3081),
        ]
    )
    data = data_parse("./src/machine_learning/data/minist")

    trainer_cfg = {
        "epochs": 10,
        "log_dir": "./logs/vae/",
        "model_dir": "./checkpoints/vae/",
        "log_interval": 10,
        "save_interval": 10,
        "batch_size": 256,
        "data_num_workers": 4,
    }

    trainer = Trainer(trainer_cfg, data, transform, vae)

    trainer.train()
    trainer.eval()


if __name__ == "__main__":
    main()
