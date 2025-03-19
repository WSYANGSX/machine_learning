from torchvision import transforms

from machine_learning.algorithms import AutoEncoder
from machine_learning.algorithms.vae import Decoder
from machine_learning.algorithms.auto_encoder.encoder import Encoder
from machine_learning.trainer import Trainer
from machine_learning.utils import data_parse


def main():
    input_size = (1, 28, 28)
    output_size = 128

    encoder = Encoder(input_size, output_size)
    decoder = Decoder(output_size)

    auto_encoder = AutoEncoder(
        "./src/machine_learning/algorithms/auto_encoder/config/config.yaml",
        {"encoder": encoder, "decoder": decoder},
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
        "log_dir": "./logs/auto_encoder/",
        "model_dir": "./checkpoints/auto_encoder/",
        "log_interval": 10,
        "save_interval": 10,
        "batch_size": 256,
        "data_num_workers": 4,
    }

    trainer = Trainer(train_cfg, data, transform, auto_encoder)

    trainer.train()
    trainer.eval()


if __name__ == "__main__":
    main()
