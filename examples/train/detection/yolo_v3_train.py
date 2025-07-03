from machine_learning.models import DarkNet53
from machine_learning.algorithms import YoloV3
from machine_learning.train import Trainer, TrainCfg
from machine_learning.utils.data_parser import ParserCfg
from machine_learning.utils.transforms import YoloTransform
from machine_learning.utils.augmentations import DEFAULT_YOLO_AUG
from machine_learning.utils.others import load_config_from_yaml


def main():
    # Step 1: Parse the data
    tfs = YoloTransform(
        augmentation=DEFAULT_YOLO_AUG,
        to_tensor=True,
        normalize=True,
        mean=[0, 0, 0],
        std=[1, 1, 1],
    )

    parser_cfg = ParserCfg(dataset_dir="./data/coco-2017", labels=True, tfs=tfs)
    data =   # (class_names, train_dataset, val_dataset)

    # Step 1: Parse configurations
    yolo_v3_cfg = load_config_from_yaml("./src/machine_learning/algorithms/detection/yolo_v3/config/yolo_v3.yaml")

    # Step 2: Build networks
    num_classes = data["class_nums"]
    default_img_size = yolo_v3_cfg["algorithm"]["default_img_size"]
    num_anchors = yolo_v3_cfg["algorithm"]["anchor_nums"]
    darknet = DarkNet53(
        default_img_shape=(3, default_img_size, default_img_size), num_anchors=num_anchors, num_classes=num_classes
    )

    # Step 2: Build the algorithm
    yolo_v3 = YoloV3(cfg=yolo_v3_cfg, data=data, models={"darknet": darknet})

    # Step 5: Configure the trainer
    trainer_cfg = TrainCfg(
        log_dir="./logs/yolov3/",
        model_dir="./checkpoints/yolov3/",
        epochs=150,
        log_interval=10,
        save_interval=10,
        save_best=True,
    )
    trainer = Trainer(trainer_cfg, yolo_v3)

    # Step 6: Train the model
    trainer.train()
    # trainer.train_from_checkpoint("/home/yangxf/WorkSpace/machine_learning/checkpoints/yolov3/checkpoint_epoch_19.pth")


if __name__ == "__main__":
    main()
