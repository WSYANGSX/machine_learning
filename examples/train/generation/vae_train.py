from torchvision import transforms

from machine_learning.algorithms import VAE
from machine_learning.networks.vae.vae_nets import Encoder, Decoder
from machine_learning.trainer import Trainer, TrainCfg
from machine_learning.utils.data_parser import ParserCfg, ParserFactory


def main():
    # Step 1: Build the network
    image_size = (1, 28, 28)
    z_dim = 64
    encoder = Encoder(image_size, z_dim)
    decoder = Decoder(z_dim)

    # Step 2: Build the algorithm
    vae = VAE(
        "./src/machine_learning/algorithms/generation/vae/config/config.yaml",
        {"encoder": encoder, "decoder": decoder},
    )

    # Step 3: Configure the augmentator/converter
    tfs = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=0.1307, std=0.3081),
        ]
    )

    # Step 4: Parse the data
    dataset_dir = "./data/minist"
    parser_cfg = ParserCfg(dataset_dir=dataset_dir, labels=True, transforms=tfs)
    parser = ParserFactory().parser_create(parser_cfg)
    dataset = parser.create()

    # Step 5: Configure the trainer
    trainer_cfg = TrainCfg(
        log_dir="./logs/vae/",
        model_dir="./checkpoints/vae/",
    )
    trainer = Trainer(trainer_cfg, dataset, vae)

    # Step 6: Train the model
    trainer.train()

    # Step 7: Evaluate the model
    vae.eval()


if __name__ == "__main__":
    main()
