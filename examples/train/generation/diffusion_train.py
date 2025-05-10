from torchvision import transforms

from machine_learning.models import UNet
from machine_learning.trainer import Trainer, TrainCfg
from machine_learning.algorithms import Diffusion
from machine_learning.utils import minist_parse


def main():
    image_size = (1, 28, 28)
    noise_predictor = UNet(image_size, 256, 1, 1, [64, 128, 256, 512], [512, 256, 128, 64])
    models = {"noise_predictor": noise_predictor}

    diffusion = Diffusion(
        "./src/machine_learning/algorithms/generation/diffusion/config/config.yaml",
        models,
    )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081]),
        ]
    )
    data = minist_parse("./data/minist")

    trainer_cfg = TrainCfg(
        dataset="minist",
        log_dir="./logs/diffusion/",
        model_dir="./checkpoints/diffusion/",
    )
    trainer = Trainer(trainer_cfg, data, transform, diffusion)

    trainer.load("/home/yangxf/my_projects/machine_learning/checkpoints/diffusion/best_model.pth")
    # trainer.train()
    trainer.eval(5)


if __name__ == "__main__":
    main()
