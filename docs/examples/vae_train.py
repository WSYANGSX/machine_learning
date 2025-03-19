from torchvision import transforms

from machine_learning.algorithms import VAE
from machine_learning.algorithms.vae import Decoder, Encoder
from machine_learning.trainer import Trainer
from machine_learning.utils import data_parse


def main():
    image_size = (1, 28, 28)
    z_dim = 60

    encoder = Encoder(image_size, z_dim)
    decoder = Decoder(z_dim)
    models = {"encoder": encoder, "decoder": decoder}

    vae = VAE(
        "./src/machine_learning/algorithms/vae/config/config.yaml",
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
        "log_dir": "./logs/vae/",
        "model_dir": "./checkpoints/vae/",
        "log_interval": 10,
        "save_interval": 10,
        "batch_size": 256,
        "data_num_workers": 4,
    }

    trainer = Trainer(train_cfg, data, transform, vae)

    trainer.train()
    trainer.eval()


if __name__ == "__main__":
    main()
