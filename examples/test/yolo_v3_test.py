from machine_learning.algorithms import YoloV3
from machine_learning.modules import DarkNet53
from machine_learning.utils.transforms import ImgTransform
from machine_learning.utils.aug_cfg import DEFAULT_YOLO_AUG
from machine_learning.utils.data_parser import YoloParserCfg, YoloParser
from machine_learning.utils.others import load_config_from_yaml


def main():
    # Step 1: Parse the data
    tfs = ImgTransform(
        augmentation=DEFAULT_YOLO_AUG["transforms"],
        bbox_params=DEFAULT_YOLO_AUG["bbox_params"],
        to_tensor=True,
        normalize=True,
        mean=[0, 0, 0],
        std=[1, 1, 1],
    )

    parser_cfg = YoloParserCfg(dataset_dir="./data/coco-2017", labels=True, tfs=tfs)
    data = YoloParser(parser_cfg).create()  # (class_names, train_dataset, val_dataset)

    # Step 1: Parse configurations
    yolo_v3_cfg = load_config_from_yaml("./src/machine_learning/algorithms/detection/yolo_v3/config/yolo_v3.yaml")

    # Step 1: Parse configurations
    num_classes = data["class_nums"]
    default_img_size = yolo_v3_cfg["algorithm"]["default_img_size"]
    num_anchors = yolo_v3_cfg["algorithm"]["anchor_nums"]
    darknet = DarkNet53(
        default_img_shape=(3, default_img_size, default_img_size), num_anchors=num_anchors, num_classes=num_classes
    )

    # Step 2: Build the algorithm
    yolo_v3 = YoloV3(cfg=yolo_v3_cfg, data=data, models={"darknet": darknet})

    # Step 3: detect
    yolo_v3.load("./checkpoints/yolov3/best_model.pth")
    yolo_v3.detect("./data/coco-2017/images/val/000000575357.jpg", default_img_size, 0.02, 0.5)


if __name__ == "__main__":
    main()
