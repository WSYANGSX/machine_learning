import torch.nn as nn
from machine_learning.models import CNN
from machine_learning.algorithms import AutoEncoder
from machine_learning.algorithms.vae.decoder import Decoder

from collections import OrderedDict


def main():
    input_size = (3, 256, 256)
    output_size = 50
    hidden_layers = OrderedDict(
        [
            ("conv1", nn.Conv2d(3, 10, 4, 2, 0)),  # (10, 127, 127)
            ("pooling1", nn.AvgPool2d(4, 2, 0)),  # (10, 62, 62)
            ("batchnorm1", nn.BatchNorm2d(10)),
            ("relu1", nn.ReLU()),
            ("conv2", nn.Conv2d(10, 5, 3, 2, 1)),  # (5, 31, 31)
            ("pooling2", nn.AvgPool2d(2, 2, 0)),  # (5,15,15)   向下取整
            ("batchnorm2", nn.BatchNorm2d(5)),
            ("relu2", nn.ReLU()),
            ("conv3", nn.Conv2d(5, 20, 2, 2, 0)),  # (20, 7, 7)
            ("pooling3", nn.AvgPool2d(2, 2, 1, ceil_mode=True)),  # (20,4,4)
            ("batchnorm3", nn.BatchNorm2d(20)),
            ("relu3", nn.ReLU()),
            ("flatten", nn.Flatten()),
            ("linear1", nn.Linear(320, 50)),
        ]
    )
    encoder = CNN(input_size=input_size, output_size=output_size, hidden_layers=hidden_layers)
    encoder.view_structure()

    decoder = Decoder(output_size)
    encoder.view_structure()
    
    auto_encoder = AutoEncoder(
        "./src/machine_learning/algorithms/auto_encoder/config/config.yaml",
        encoder,
        decoder,
        "cuda",
    )

    auto_encoder.train_model()

    auto_encoder.visualize_reconstruction()


if __name__ == "__main__":
    main()
