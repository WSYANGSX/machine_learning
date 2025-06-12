from machine_learning.algorithms import YoloV3
from machine_learning.models import Darknet, FPN
from machine_learning.train import Trainer, TrainCfg
from machine_learning.utils import load_config_from_path
from machine_learning.utils.transforms import YoloTransform
from machine_learning.utils.dataload import ParserCfg, ParserFactory


def main():
    # 配置模型和算法
    yolo_v3_cfg = load_config_from_path("./src/machine_learning/algorithms/detection/yolo_v3/config/yolo_v3.yaml")
    imgae_size = yolo_v3_cfg["algorithm"]["image_size"]
    num_classes = yolo_v3_cfg["algorithm"]["num_classes"]
    num_anchors = yolo_v3_cfg["algorithm"]["num_anchors"]

    # 构建算法
    darknet = Darknet(imgae_size)
    fpn = FPN(num_anchors, num_classes)
    yolo_v3 = YoloV3(cfg=yolo_v3_cfg, models={"darknet": darknet, "fpn": fpn})

    # 配置transform
    tfs = YoloTransform(augmentation="default", mean=[0.471, 0.448, 0.408], std=[0.234, 0.239, 0.242])

    # 加载数据
    dataset_dir = "./data/coco-2017"
    parser_cfg = ParserCfg(dataset_dir=dataset_dir, labels=True, transforms=tfs)
    parser = ParserFactory().parser_create(parser_cfg)
    datasets = parser.create()

    # 配置训练参数
    trainer_cfg = TrainCfg(
        log_dir="./logs/yolov3/",
        model_dir="./checkpoints/yolov3/",
        batch_size=5,
        data_num_workers=8,
        epochs=500,
        log_interval=20,
        save_interval=20,
        save_best=True,
    )
    trainer = Trainer(trainer_cfg, datasets, yolo_v3)

    # 模型训练
    trainer.train()
    # trainer.load("/home/yangxf/my_projects/machine_learning/checkpoints/auto_encoder/best_model.pth")
    trainer.eval()


if __name__ == "__main__":
    main()
