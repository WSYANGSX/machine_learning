from machine_learning.networks.yolo import V13Net
from machine_learning.algorithms.detection import YoloV13
from machine_learning.trainer import Trainer, TrainCfg
from machine_learning.utils.aug import DEFAULT_YOLO_AUG
from machine_learning.utils.transforms import ImgTransform
from machine_learning.dataset.parsers import YoloParser, YoloParserCfg
from machine_learning.utils import load_config_from_yaml


def main():
    # Step 1: Parse the data
    tfs = ImgTransform(
        aug_cfg=DEFAULT_YOLO_AUG,
        normalize=True,
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_tensor=True,
    )

    parser_cfg = YoloParserCfg(
        dataset_dir="/home/yangxf/WorkSpace/datasets/..datasets/coco",
        labels=True,
        tfs=tfs,
        multiscale=False,
        img_size=640,
    )
    data = YoloParser(parser_cfg).create()  # (class_names, train_dataset, val_dataset)

    # Step 1: Parse configurations
    cfg = load_config_from_yaml("./src/machine_learning/algorithms/detection/yolo_v13/config/yolo_v13.yaml")

    # Step 2: Build networks
    num_classes = data["class_nums"]
    img_size = 640
    net = V13Net(img_shape=(3, img_size, img_size), nc=num_classes, scale="n")

    # Step 2: Build the algorithm
    yolo_v13 = YoloV13(cfg=cfg, net=net, data=data)

    # Step 5: Configure the trainer
    trainer_cfg = TrainCfg(
        log_dir="./logs/yolo_v13/",
        ckpt_dir="./checkpoints/yolo_v13/",
        epochs=600,
        log_interval=10,
        save_interval=10,
        save_best=True,
    )
    trainer = Trainer(trainer_cfg, yolo_v13)

    # Step 6: Train the model
    # trainer.train()
    trainer.train_from_checkpoint(
        "/home/yangxf/WorkSpace/machine_learning/checkpoints/yolo_v13/2025-09-05_21-49/checkpoint_epoch_99.pth"
    )


if __name__ == "__main__":
    main()
