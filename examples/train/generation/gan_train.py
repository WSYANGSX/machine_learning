from torchvision import transforms

from machine_learning.algorithms import GAN
from machine_learning.models import Generator, Discriminator
from machine_learning.train import Trainer, TrainCfg
from machine_learning.utils.dataload import ParserCfg, ParserFactory


def main():
    # Step 1: Build the network
    generator_input_dim = 100
    image_size = (1, 28, 28)
    generator = Generator(generator_input_dim, image_size)
    discriminator = Discriminator(image_size)

    # Step 2: Build the algorithm
    gan = GAN(
        "./src/machine_learning/algorithms/generation/gan/config/config.yaml",
        {"generator": generator, "discriminator": discriminator},
    )

    # Step 3: Configure the augmentator/converter
    tfs = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081]),
        ]
    )

    # Step 4: Parse the data
    dataset_dir = "./data/minist"
    parser_cfg = ParserCfg(dataset_dir=dataset_dir, labels=True, transforms=tfs)
    parser = ParserFactory().parser_create(parser_cfg)
    dataset = parser.create()

    # Step 5: Configure the trainer
    trainer_cfg = TrainCfg(
        log_dir="./logs/gan/",
        model_dir="./checkpoints/gan/",
    )
    trainer = Trainer(trainer_cfg, dataset, gan)

    # Step 6: Train the model
    trainer.train()

    # Step 7: Evaluate the model
    # gan.load("/home/yangxf/my_projects/machine_learning/checkpoints/gan/checkpoint_epoch_999.pth")
    gan.eval(16)


if __name__ == "__main__":
    main()
