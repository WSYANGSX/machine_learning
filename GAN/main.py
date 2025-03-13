from collections import OrderedDict

import torch.nn as nn
from GAN import GAN
from models import Generator, Discriminator
from torch.nn.utils import spectral_norm


def main():
    g_input_dim = 100
    g_layers = OrderedDict(
        [
            ("lin1", nn.Linear(g_input_dim, 128)),
            ("LRelu1", nn.LeakyReLU(0.2)),
            ("lin2", nn.Linear(128, 256)),
            ("BatchNorm2", nn.BatchNorm1d(256)),
            ("LRelu2", nn.LeakyReLU(0.2)),
            ("lin3", nn.Linear(256, 512)),
            ("BatchNorm3", nn.BatchNorm1d(512)),
            ("LRelu3", nn.LeakyReLU(0.2)),
            ("lin4", nn.Linear(512, 1024)),
            ("BatchNorm4", nn.BatchNorm1d(1024)),
            ("LRelu4", nn.LeakyReLU(0.2)),
            ("lin5", nn.Linear(1024, 28 * 28)),
            ("Tanh", nn.Tanh()),
            ("reshape", nn.Unflatten(1, (1, 28, 28))),
        ]
    )
    generator = Generator(input_dim=g_input_dim, layers=g_layers)

    d_input_size = (1, 1, 28, 28)
    d_layers = OrderedDict(
        [
            ("reshape", nn.Flatten()),
            ("lin1", nn.Linear(28 * 28, 512)),
            ("LRelu1", nn.LeakyReLU(0.2)),
            ("lin2", nn.Linear(512, 256)),
            ("LRelu2", nn.LeakyReLU(0.2)),
            ("lin3", nn.Linear(256, 1)),
            ("sigmoid", nn.Sigmoid()),
        ]
    )
    discriminator = Discriminator(input_size=d_input_size, layers=d_layers)

    gan = GAN(
        config_file="./Machine learning/GAN/config.yaml",
        input_dim=g_input_dim,
        generator=generator,
        discriminator=discriminator,
        device="cuda",
    )

    gan.train_model()

    gan.visualize_reconstruction()


if __name__ == "__main__":
    main()
