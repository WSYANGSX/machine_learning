from torchvision import transforms
from machine_learning.models import UNet
from machine_learning.train import Trainer, TrainCfg
from machine_learning.algorithms import Diffusion
from machine_learning.utils.dataload import ParserFactory, ParserCfg


def main():
    # Step 1: Configure the augmentator/converter and parse data
    tfs = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081]),
        ]
    )

    dataset_dir = "./data/minist"
    parser_cfg = ParserCfg(dataset_dir=dataset_dir, labels=True, tfs=tfs)
    parser = ParserFactory().create_parser(parser_cfg)
    data = parser.create()

    # Step 2: Build the network
    image_size = (1, 28, 28)
    noise_predictor = UNet(image_size, 256, 1, 1, [64, 128, 256, 512], [512, 256, 128, 64])

    # Step 3: Build the algorithm
    diffusion = Diffusion(
        "./src/machine_learning/algorithms/generation/diffusion/config/config.yaml",
        {"noise_predictor": noise_predictor},
        data,
    )

    # Step 5: Configure the trainer
    trainer_cfg = TrainCfg(
        log_dir="./logs/diffusion/",
        model_dir="./checkpoints/diffusion/",
    )
    trainer = Trainer(trainer_cfg, diffusion)

    # Step 6: Train the model
    trainer.train()

    # Step 6: Evaluate the model
    # diffusion.load("/home/yangxf/my_projects/machine_learning/checkpoints/diffusion/best_model.pth")
    diffusion.eval(5)


if __name__ == "__main__":
    main()
