import numpy as np
import torch.nn as nn
from torchvision import transforms
from torchinfo import summary

from machine_learning.algorithms import Diffusion
from machine_learning.trainer import Trainer
from machine_learning.models import BaseNet
from machine_learning.utils import data_parse


# 模型定义
class UNet(BaseNet):
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


def main():
    image_size = (1, 28, 28)

    noise_predicter = UNet(image_size)
    models = {"noise_predicter": noise_predicter}

    diffusion = Diffusion(
        "./src/machine_learning/algorithms/diffusion/config/config.yaml",
        models,
    )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081]),
        ]
    )
    data = data_parse("./src/machine_learning/data/minist")

    train_cfg = {
        "epochs": 50,
        "log_dir": "./logs/diffusion/",
        "model_dir": "./checkpoints/diffusion/",
        "log_interval": 10,
        "save_interval": 10,
        "batch_size": 512,
        "data_num_workers": 4,
    }

    trainer = Trainer(train_cfg, data, transform, diffusion)

    # trainer.load("/home/yangxf/my_projects/machine_learning/checkpoints/gan/checkpoint_epoch_999.pth")
    trainer.train()
    trainer.eval(16)


if __name__ == "__main__":
    main()
