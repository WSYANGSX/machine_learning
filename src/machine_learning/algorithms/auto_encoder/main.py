import torch.nn as nn

from collections import OrderedDict
from auto_encoder import AutoEncoder


def main():
    # input image size N*1*28*28
    encoder_layers = OrderedDict(
        [
            (
                "conv1",
                nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=2, padding=1),
            ),
            (
                "BatchNorm1",
                nn.BatchNorm2d(3),
            ),
            ("relu1", nn.ReLU()),  # output image size N*3*14*14
            (
                "conv2",
                nn.Conv2d(3, 6, 3, 2, 1),
            ),
            (
                "BatchNorm2",
                nn.BatchNorm2d(6),
            ),
            ("relu2", nn.ReLU()),  # output image size N*6*7*7
            ("reshape", nn.Flatten()),  # output image size N*294
            ("linear", nn.Linear(294, 128)),  # output image size N*128
        ]
    )
    decoder_layers = OrderedDict(
        [
            ("linear", nn.Linear(128, 294)),
            ("reshape", nn.Unflatten(1, (6, 7, 7))),
            ("BatchNorm1", nn.BatchNorm2d(6)),
            ("relu1", nn.ReLU()),
            (
                "deconv1",
                nn.ConvTranspose2d(6, 3, 3, 2, 1, output_padding=1),
            ),  # 输出尺寸：(16,12,12)),
            ("BatchNorm2", nn.BatchNorm2d(3)),
            ("relu2", nn.ReLU()),
            (
                "deconv2",
                nn.ConvTranspose2d(3, 1, 3, 2, 1, output_padding=1),
            ),  # 输出尺寸：(16,12,12)),
            ("relu3", nn.ReLU()),
        ]
    )
    auto_encoder = AutoEncoder(
        "./src/machine_learning/algorithm/auto_encoder/config/config.yaml",
        encoder_layers,
        decoder_layers,
        "cuda",
    )

    auto_encoder.train_model()

    auto_encoder.visualize_reconstruction()


if __name__ == "__main__":
    main()
