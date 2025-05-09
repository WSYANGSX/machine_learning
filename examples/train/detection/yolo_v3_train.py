from torchvision import transforms

from machine_learning.trainer import Trainer
from machine_learning.algorithms import YoloV3
from machine_learning.models import Darknet, FPN
from machine_learning.utils import load_config_from_path, yolo_parse


def main():
    # 配置模型和算法
    yolo_v3_cfg = load_config_from_path("./src/machine_learning/algorithms/detection/yolo_v3/config/yolo_v3.yaml")
    imgae_size = yolo_v3_cfg["algorithm"]["image_size"]
    num_classes = yolo_v3_cfg["algorithm"]["num_classes"]
    num_anchors = yolo_v3_cfg["algorithm"]["num_anchors"]

    darknet = Darknet(imgae_size)
    fpn = FPN(num_anchors, num_classes)
    yolo_v3 = YoloV3(cfg=yolo_v3_cfg, models={"darknet": darknet, "fpn": fpn})

    # 加载数据
    dataset = yolo_parse("/home/yangxf/WorkSpace/machine_learning/data/coco-2017")

    # 配置transform
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081]),
        ]
    )

    # 配置训练参数
    trainer_cfg = {
        "epochs": 100,
        "log_dir": "./logs/yolo_v3/",
        "model_dir": "./checkpoints/yolo_v3/",
        "log_interval": 10,
        "save_interval": 10,
        "batch_size": 256,
        "data_num_workers": 4,
    }
    trainer = Trainer(trainer_cfg, dataset, transform, yolo_v3)

    # 模型训练
    trainer.train()
    # trainer.load("/home/yangxf/my_projects/machine_learning/checkpoints/auto_encoder/best_model.pth")
    trainer.eval()


if __name__ == "__main__":
    main()
