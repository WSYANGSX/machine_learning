import torch.nn as nn

from collections import OrderedDict
from machine_learning.algorithms import AutoEncoder
from machine_learning.models import CNN, MLP


def main():
    image_size = (1, 28, 28)
    
    encoder = CNN
    
    encoder_layers = OrderedDict(
        [
            (
                "conv1",
                nn.Conv2d(1, 3, 3, 2, 1),
            ),
            (
                "BatchNorm1",
                nn.BatchNorm2d(3),
            ),
            ("relu1", nn.ReLU()),
            (
                "conv2",
                nn.Conv2d(3, 6, 3, 2, 1),
            ),
            (
                "BatchNorm2",
                nn.BatchNorm2d(6),
            ),
            ("relu2", nn.ReLU()),
            ("reshape", nn.Flatten()),
            ("linear", nn.Linear(294, 128)),
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
        "./src/machine_learning/algorithms/auto_encoder/config/config.yaml",
        encoder_layers,
        decoder_layers,
        "cuda",
    )

    auto_encoder.train_model()

    auto_encoder.visualize_reconstruction()


if __name__ == "__main__":
    main()
