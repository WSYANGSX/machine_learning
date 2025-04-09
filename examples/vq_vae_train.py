import torch
import torch.nn as nn
from torchinfo import summary
from torchvision import transforms

from machine_learning.trainer import Trainer
from machine_learning.algorithms import VQ_VAE
from machine_learning.models import BaseNet, ResidualBlock2D
from machine_learning.utils import data_parse, cal_conv_output_size, cal_convtrans_output_size


# 模型定义
class Encoder(BaseNet):
    def __init__(self, input_size: tuple[int], output_size: tuple[int]):
        super().__init__()

        self.input_size = input_size
        self.in_channels = self.input_size[0]
        self.output_size = output_size
        self.out_channels = self.output_size[0]

        self.conv1 = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.out_channels // 4, kernel_size=4, stride=2, padding=1
        )
        output_size = cal_conv_output_size(self.input_size, self.conv1)

        self.conv2 = nn.Conv2d(
            in_channels=self.out_channels // 4, out_channels=self.out_channels // 2, kernel_size=4, stride=2, padding=1
        )
        output_size = cal_conv_output_size(output_size, self.conv2)

        self.conv3 = nn.Conv2d(
            in_channels=self.out_channels // 2, out_channels=self.out_channels, kernel_size=3, stride=2, padding=1
        )
        output_size = cal_conv_output_size(output_size, self.conv3)

        self.residual1 = ResidualBlock2D(in_channels=self.out_channels, out_channels=self.out_channels, dropout=0)

        if output_size != self.output_size:
            raise ValueError("Encoder output size is not as excepted.")

        self.act = nn.SiLU()

    def forward(self, inputs: torch):
        x = self.conv1(inputs)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.act(x)
        x = self.residual1(x)

        return x

    def view_structure(self):
        summary(self, (1, *self.input_size))


class Decoder(BaseNet):
    def __init__(self, input_size: tuple[int], output_size: tuple[int]):
        super().__init__()

        self.input_size = input_size
        self.in_channels = self.input_size[0]
        self.output_size = output_size
        self.out_channels = self.output_size[0]

        self.conv1 = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.in_channels // 2, kernel_size=3, stride=1, padding=1
        )
        output_size = cal_conv_output_size(self.input_size, self.conv1)

        self.residual1 = ResidualBlock2D(
            in_channels=self.in_channels // 2, out_channels=self.in_channels // 2, dropout=0
        )

        self.conv_trans1 = nn.ConvTranspose2d(
            in_channels=self.in_channels // 2, out_channels=self.in_channels // 4, kernel_size=4, stride=2, padding=1
        )
        output_size = cal_convtrans_output_size(output_size, self.conv_trans1)

        self.conv_trans2 = nn.ConvTranspose2d(
            in_channels=self.in_channels // 4, out_channels=self.out_channels, kernel_size=2, stride=2, padding=1
        )
        output_size = cal_convtrans_output_size(output_size, self.conv_trans2)

        self.conv_trans3 = nn.ConvTranspose2d(
            in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=4, stride=2, padding=1
        )
        output_size = cal_convtrans_output_size(output_size, self.conv_trans3)

        self.act = nn.SiLU()

        if output_size != self.output_size:
            raise ValueError("Decoder output size is not as excepted.")

    def forward(self, inputs: torch.Tensor):
        x = self.conv1(inputs)
        x = self.act(x)

        x = self.residual1(x)
        x = self.act(x)

        x = self.conv_trans1(x)
        x = self.act(x)

        x = self.conv_trans2(x)
        x = self.act(x)

        x = self.conv_trans3(x)

        return x

    def view_structure(self):
        summary(self, (1, *self.input_size))


def main():
    image_size = (1, 28, 28)
    latent_size = (64, 4, 4)

    encoder = Encoder(input_size=image_size, output_size=latent_size)
    decoder = Decoder(input_size=latent_size, output_size=image_size)
    models = {"encoder": encoder, "decoder": decoder}

    vq_vae = VQ_VAE(
        "./src/machine_learning/algorithms/vq_vae/config/config.yaml",
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
        "log_dir": "./logs/vq_vae/",
        "model_dir": "./checkpoints/vq_vae/",
        "log_interval": 10,
        "save_interval": 10,
        "batch_size": 256,
        "data_num_workers": 4,
    }

    trainer = Trainer(train_cfg, data, transform, vq_vae)

    trainer.train()
    trainer.eval()


if __name__ == "__main__":
    main()
