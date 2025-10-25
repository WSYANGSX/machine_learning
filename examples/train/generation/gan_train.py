from machine_learning.algorithms import GAN
from machine_learning.networks import Generator, Discriminator
from machine_learning.trainer import Trainer, TrainCfg
from examples.transforms import ImgTransform
from examples.transforms import DEFAULT_AUG
from machine_learning.dataset.parsers import ParserCfg, MinistParser


def main():
    # Step 1: Configure the augmentator/converter and parse the data
    tfs = ImgTransform(aug_cfg=DEFAULT_AUG, normalize=True, mean=[0.1307], std=[0.3081], to_tensor=True)

    dataset_dir = "./data/minist"
    parser_cfg = ParserCfg(dataset_dir=dataset_dir, labels=True, tfs=tfs)
    parser = MinistParser(parser_cfg)
    data = parser.create()

    # Step 2: Build the network
    image_shape = data["image_shape"]
    generator_input_dim = 100
    generator = Generator(generator_input_dim, image_shape)
    discriminator = Discriminator(image_shape)

    # Step 2: Build the algorithm
    gan = GAN(
        "./src/machine_learning/algorithms/generation/gan/config/config.yaml",
        generator,
        discriminator,
    )

    # Step 5: Configure the trainer
    trainer_cfg = TrainCfg(
        log_dir="./logs/gan/",
        model_dir="./checkpoints/gan/",
    )
    trainer = Trainer(trainer_cfg, gan, data)

    # Step 6: Train the model
    trainer.train()

    # Step 7: Evaluate the model
    # gan.load("/home/yangxf/my_projects/machine_learning/checkpoints/gan/checkpoint_epoch_999.pth")
    gan.eval(16)


if __name__ == "__main__":
    main()
