import torch.nn as nn
from torchvision import transforms

from machine_learning.models import CNN
from machine_learning.algorithms import AutoEncoder
from machine_learning.algorithms.vae.decoder import Decoder
from machine_learning.utils import data_parse


from collections import OrderedDict


def main():
    input_size = (1, 28, 28)
    output_size = 128

    hidden_layers = OrderedDict(
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
    encoder = CNN(input_size=input_size, output_size=output_size, hidden_layers=hidden_layers)
    encoder.view_structure()

    decoder = Decoder(output_size)
    encoder.view_structure()

    auto_encoder = AutoEncoder(
        "./src/machine_learning/algorithms/auto_encoder/config/config.yaml",
        "cuda",
    )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=0.1307, std=0.3081),
        ]
    )
    data = data_parse("./src/machine_learning/data/minist")
    auto_encoder._load_datasets(*data, transform)

    auto_encoder.train_model()

    auto_encoder.visualize_reconstruction()


if __name__ == "__main__":
    main()
