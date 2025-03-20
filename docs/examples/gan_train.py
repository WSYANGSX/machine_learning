import torch.nn as nn
from torchvision import transforms
from torchinfo import summary

from machine_learning.algorithms import GAN
from machine_learning.trainer import Trainer
from machine_learning.models import BaseNet
from machine_learning.utils import data_parse


# 模型定义
class Generator(BaseNet):
    def __init__(self, input_dim: int) -> None:
        """
        Network for gan generator.

        Args:
            input_dim (int): 输入特征向量的维度.
        """
        super().__init__()

        self.input_dim = input_dim

        self.layers = self.layer1 = nn.Sequential(nn.Linear(self.input_dim, 294), nn.Unflatten(1, (6, 7, 7)))

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
        out_put = self.layer3(mid_val)

        return out_put

    def view_structure(self):
        summary(self, input_size=(1, self.input_dim))


class Discriminator(BaseNet):
    def __init__(self, input_size: tuple[int]) -> None:
        """
        Network for gan discrimator.

        Args:
            input_size (tuple[int]): 输入数据的size (channels, ...).
        """
        super().__init__()

        self.input_size = input_size

        self.layer1 = nn.Sequential(nn.Conv2d(1, 3, 2, 2, 0), nn.BatchNorm2d(3), nn.ReLU())  # (3,14,14)
        self.layer2 = nn.Sequential(nn.Conv2d(3, 10, 2, 2, 0), nn.BatchNorm2d(10), nn.ReLU())  # (10,7,7)
        self.layer3 = nn.Sequential(nn.Conv2d(10, 15, 2, 2, 0), nn.BatchNorm2d(15), nn.ReLU(), nn.Flatten())  # (15,3,3)
        self.layer4 = nn.Sequential(nn.Linear(135, 1), nn.Sigmoid())

    def forward(self, x):
        mid_val = self.layer1(x)
        mid_val = self.layer2(mid_val)
        mid_val = self.layer3(mid_val)
        output = self.layer4(mid_val)

        return output

    def view_structure(self):
        summary(self, input_size=(1, *self.input_size))


def main():
    generator_input_dim = 64
    discriminator_input_size = (1, 28, 28)

    generator = Generator(generator_input_dim)
    discriminator = Discriminator(discriminator_input_size)
    models = {"generator": generator, "discriminator": discriminator}

    gan = GAN(
        "./src/machine_learning/algorithms/gan/config/config.yaml",
        models,
    )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=0.1307, std=0.3081),
        ]
    )
    data = data_parse("./src/machine_learning/data/minist")

    train_cfg = {
        "epochs": 100,
        "log_dir": "./logs/gan/",
        "model_dir": "./checkpoints/gan/",
        "log_interval": 10,
        "save_interval": 10,
        "batch_size": 256,
        "data_num_workers": 4,
    }

    trainer = Trainer(train_cfg, data, transform, gan)

    trainer.load("/home/yangxf/my_projects/machine_learning/checkpoints/gan/checkpoint_epoch_99.pth")
    trainer.eval()


if __name__ == "__main__":
    main()
