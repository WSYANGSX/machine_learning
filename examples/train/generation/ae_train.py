import torch.nn as nn
from torchvision import transforms

from machine_learning.models import BaseNet
from machine_learning.algorithms import AutoEncoder
from machine_learning.train import Trainer, TrainCfg
from machine_learning.utils import ParserFactory, ParserCfg


class Encoder(BaseNet):
    def __init__(self, input_size: tuple[int], z_dim: int) -> None:
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
        Network for vae decoder.

        Args:
            z_dim (int): 多维高斯的维度.
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


def main():
    # 步骤1：构建网络
    input_size = (1, 28, 28)
    output_size = 128
    encoder = Encoder(input_size, output_size)
    decoder = Decoder(output_size)

    # 步骤2：构建算法
    auto_encoder = AutoEncoder(
        "./src/machine_learning/algorithms/generation/auto_encoder/config/config.yaml",
        {"encoder": encoder, "decoder": decoder},
    )

    # 步骤3：配置增强器和转换器
    tfs = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081]),
        ]
    )

    # 步骤4：解析数据
    dataset_dir = "./data/minist"
    parser_cfg = ParserCfg(dataset_dir=dataset_dir, labels=True, data_load_method="full", transforms=tfs)
    parser = ParserFactory().parser_create(parser_cfg)
    dataset = parser.create()

    # 步骤5：配置训练器
    trainer_cfg = TrainCfg(
        log_dir="./logs/auto_encoder/",
        model_dir="./checkpoints/auto_encoder/",
    )
    trainer = Trainer(trainer_cfg, dataset, auto_encoder)

    # 步骤6：训练评估模型
    trainer.train()
    # trainer.load("/home/yangxf/my_projects/machine_learning/checkpoints/auto_encoder/best_model.pth")
    trainer.eval()


if __name__ == "__main__":
    main()
