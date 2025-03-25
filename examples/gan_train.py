import numpy as np
import torch.nn as nn
from torchvision import transforms
from torchinfo import summary

from machine_learning.algorithms import GAN
from machine_learning.trainer import Trainer
from machine_learning.models import BaseNet
from machine_learning.utils import data_parse


# 模型定义
class Generator(BaseNet):
    def __init__(self, input_dim: int, output_size: tuple[int]) -> None:
        """
        Network for gan generator.

        Args:
            input_dim (int): 输入特征向量的维度.
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

        self.layer1 = nn.Sequential(nn.Flatten())
        self.layer2 = nn.Sequential(nn.Linear(np.prod(input_size), 1024), nn.LeakyReLU(0.2), nn.Dropout(0.3))
        self.layer3 = nn.Sequential(nn.Linear(1024, 512), nn.LeakyReLU(0.2), nn.Dropout(0.3))
        self.layer4 = nn.Sequential(nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3))
        self.layer5 = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())

    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        output = self.layer5(y)

        return output

    def view_structure(self):
        summary(self, input_size=(1, *self.input_size))


def main():
    generator_input_dim = 100
    image_size = (1, 28, 28)

    generator = Generator(generator_input_dim, image_size)
    discriminator = Discriminator(image_size)
    models = {"generator": generator, "discriminator": discriminator}

    gan = GAN(
        "./src/machine_learning/algorithms/gan/config/config.yaml",
        models,
    )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    data = data_parse("./src/machine_learning/data/minist")

    train_cfg = {
        "epochs": 50,
        "log_dir": "./logs/gan/",
        "model_dir": "./checkpoints/gan/",
        "log_interval": 10,
        "save_interval": 10,
        "batch_size": 128,
        "data_num_workers": 4,
    }

    trainer = Trainer(train_cfg, data, transform, gan)

    # trainer.load("/home/yangxf/my_projects/machine_learning/checkpoints/gan/checkpoint_epoch_999.pth")
    trainer.train()
    trainer.eval(16)


if __name__ == "__main__":
    main()
