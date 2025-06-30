from machine_learning.algorithms import YoloV3
from machine_learning.models import Darknet, FPN
from machine_learning.train import Trainer, TrainCfg
from machine_learning.utils.transforms import YoloTransform
from machine_learning.utils.augmentations import DEFAULT_YOLO_AUG
from machine_learning.utils.others import load_config_from_yaml
from machine_learning.utils.dataload import ParserCfg, ParserFactory


def main():
    # Step 1: Parse the data
    tfs = YoloTransform(
        augmentation=DEFAULT_YOLO_AUG,
        to_tensor=True,
        normalize=True,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    dataset_dir = "./data/coco-2017"
    parser_cfg = ParserCfg(dataset_dir=dataset_dir, labels=True, tfs=tfs)
    parser = ParserFactory().create_parser(parser_cfg)
    data = parser.create()  # (class_names, train_dataset, val_dataset)

    # Step 1: Parse configurations
    dataset_cfg = load_config_from_yaml("./data/coco-2017/metadata.yaml")
    yolo_v3_cfg = load_config_from_yaml("./src/machine_learning/algorithms/detection/yolo_v3/config/yolo_v3.yaml")

    # Step 2: Build networks
    class_nums = dataset_cfg["class_nums"]
    default_image_size = yolo_v3_cfg["algorithm"]["default_img_size"]
    anchor_nums = yolo_v3_cfg["algorithm"]["anchor_nums"]
    darknet = Darknet(default_image_size)
    fpn = FPN(anchor_nums, class_nums)

    # Step 2: Build the algorithm
    yolo_v3 = YoloV3(cfg=yolo_v3_cfg, data=data, models={"darknet": darknet, "fpn": fpn})

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
    # trainer.train()
    trainer.train_from_checkpoint("/home/yangxf/WorkSpace/machine_learning/checkpoints/yolov3/checkpoint_epoch_19.pth")

    # # Step 7: eval
    # yolo_v3.load("/home/yangxf/WorkSpace/machine_learning/checkpoints/yolov3/best_model.pth")
    # yolo_v3.eval("/home/yangxf/WorkSpace/machine_learning/data/coco-2017/images/test/000000000001.jpg")


if __name__ == "__main__":
    main()
