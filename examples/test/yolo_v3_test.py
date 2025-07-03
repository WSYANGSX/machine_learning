from machine_learning.algorithms import YoloV3
from machine_learning.models import DarkNet53
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
    yolo_v3.detect("./data/coco-2017/images/test/000000581067.jpg", default_img_size, 0.04, 0.5)


if __name__ == "__main__":
    main()
