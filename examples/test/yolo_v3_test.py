from machine_learning.algorithms import YoloV3
from machine_learning.models import Darknet, FPN
from machine_learning.utils.transforms import YoloTransform
from machine_learning.utils.others import load_config_from_yaml
from machine_learning.utils.dataload import ParserCfg, ParserFactory


def main():
    # Step 3: Configure the augmentator/converter
    tfs = YoloTransform(mean=[0.471, 0.448, 0.408], std=[0.234, 0.239, 0.242])

    # Step 4: Parse the data
    dataset_dir = "./data/coco-2017"
    parser_cfg = ParserCfg(dataset_dir=dataset_dir, labels=True, tfs=tfs)
    parser = ParserFactory().create_parser(parser_cfg)
    data = parser.create()  # (class_names, train_dataset, val_dataset)

    # Step 1: Parse configurations
    dataset_cfg = load_config_from_yaml("./data/coco-2017/metadata.yaml")
    yolo_v3_cfg = load_config_from_yaml("./src/machine_learning/algorithms/detection/yolo_v3/config/yolo_v3.yaml")

    # Step 2: Build networks
    class_nums = dataset_cfg["class_nums"]
    image_size = yolo_v3_cfg["algorithm"]["img_size"]
    anchor_nums = yolo_v3_cfg["algorithm"]["anchor_nums"]
    darknet = Darknet((3, image_size, image_size))
    fpn = FPN(anchor_nums, class_nums)

    # Step 2: Build the algorithm
    yolo_v3 = YoloV3(cfg=yolo_v3_cfg, data=data, models={"darknet": darknet, "fpn": fpn})

    # Step 3: detect
    yolo_v3.load("./checkpoints/yolov3/checkpoint_epoch_39.pth")
    yolo_v3.detect("./data/coco-2017/images/val/000000570782.jpg")


if __name__ == "__main__":
    main()
