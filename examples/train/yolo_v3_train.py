from torchvision import transforms

from machine_learning.trainer import Trainer
from machine_learning.algorithms import YoloV3
from machine_learning.models import Darknet, FPN


def main():
    input_size = (3, 416, 416)

    darknet = Darknet(input_size)
    fpn = FPN()

    yolo_v3 = YoloV3(
        "./src/machine_learning/algorithms/detection/yolo_v3/config/yolo_v3.yaml",
        {"darknet": darknet, "fpn": fpn},
    )

    data = data_parse("./src/machine_learning/data/minist")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081]),
        ]
    )

    train_cfg = {
        "epochs": 100,
        "log_dir": "./logs/yolo_v3/",
        "model_dir": "./checkpoints/yolo_v3/",
        "log_interval": 10,
        "save_interval": 10,
        "batch_size": 256,
        "data_num_workers": 4,
    }

    trainer = Trainer(train_cfg, data, transform, yolo_v3)

    trainer.train()
    # trainer.load("/home/yangxf/my_projects/machine_learning/checkpoints/auto_encoder/best_model.pth")
    trainer.eval()


if __name__ == "__main__":
    main()
