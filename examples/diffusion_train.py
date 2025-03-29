from torchvision import transforms

from machine_learning.algorithms import Diffusion
from machine_learning.trainer import Trainer
from machine_learning.models import UNet
from machine_learning.utils import data_parse


def main():
    input_size = (1, 28, 28)
    noise_predictor = UNet(input_size, 256, 1, 1, [64, 128, 256, 512], [512, 256, 128, 64])
    models = {"noise_predictor": noise_predictor}

    diffusion = Diffusion(
        "./src/machine_learning/algorithms/diffusion/config/config.yaml",
        models,
    )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081]),
        ]
    )
    data = data_parse("./src/machine_learning/data/minist")

    train_cfg = {
        "epochs": 50,
        "log_dir": "./logs/diffusion/",
        "model_dir": "./checkpoints/diffusion/",
        "log_interval": 10,
        "save_interval": 10,
        "batch_size": 512,
        "data_num_workers": 4,
    }

    trainer = Trainer(train_cfg, data, transform, diffusion)

    trainer.load("/home/yangxf/my_projects/machine_learning/checkpoints/diffusion/checkpoint_epoch_19.pth")
    # trainer.train()
    trainer.eval(16)


if __name__ == "__main__":
    main()
