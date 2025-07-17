from torchvision import transforms

from machine_learning.algorithms import VQ_VAE
from machine_learning.networks.vq_vae.vq_vae_nets import Encoder, Decoder
from machine_learning.train import Trainer, TrainCfg
from machine_learning.utils.data_parser import ParserCfg, ParserFactory


def main():
    # Step 1: Build the network
    image_size = (1, 28, 28)
    latent_size = (64, 7, 7)
    encoder = Encoder(input_size=image_size, output_size=latent_size)
    decoder = Decoder(input_size=latent_size, output_size=image_size)

    # Step 2: Build the algorithm
    vq_vae = VQ_VAE(
        "./src/machine_learning/algorithms/generation/vq_vae/config/config.yaml",
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
        log_dir="./logs/vq_vae/",
        model_dir="./checkpoints/vq_vae/",
    )
    trainer = Trainer(trainer_cfg, dataset, vq_vae)

    # Step 6: Train the model
    trainer.train()

    # Step 7: Evaluate the model
    vq_vae.eval()


if __name__ == "__main__":
    main()
