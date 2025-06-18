from machine_learning.algorithms import YoloV3
from machine_learning.models import Darknet, FPN
from machine_learning.train import Trainer, TrainCfg
from machine_learning.utils.transforms import YoloTransform
from machine_learning.utils.others import load_config_from_yaml
from machine_learning.utils.dataload import ParserCfg, ParserFactory


def main():
    # Step 1: Build the network
    class_nums = 80
    yolo_v3_cfg = load_config_from_yaml("./src/machine_learning/algorithms/detection/yolo_v3/config/yolo_v3.yaml")
    default_image_size = yolo_v3_cfg["algorithm"]["default_img_size"]
    anchor_nums = yolo_v3_cfg["algorithm"]["anchor_nums"]
    darknet = Darknet(default_image_size)
    fpn = FPN(anchor_nums, class_nums)

    # Step 2: Build the algorithm
    yolo_v3 = YoloV3(cfg=yolo_v3_cfg, models={"darknet": darknet, "fpn": fpn})

    # Step 3: Configure the augmentator/converter
    tfs = YoloTransform(augmentation="default", mean=[0.471, 0.448, 0.408], std=[0.234, 0.239, 0.242])

    # Step 4: Parse the data
    dataset_dir = "./data/coco-2017"
    parser_cfg = ParserCfg(dataset_dir=dataset_dir, labels=True, transforms=tfs)
    parser = ParserFactory().parser_create(parser_cfg)
    data = parser.create()  # (class_names, train_dataset, val_dataset)

    # Step 5: Configure the trainer
    trainer_cfg = TrainCfg(
        log_dir="./logs/yolov3/",
        model_dir="./checkpoints/yolov3/",
        batch_size=64,
        data_num_workers=8,
        epochs=500,
        log_interval=10,
        save_interval=10,
        save_best=True,
    )
    trainer = Trainer(trainer_cfg, data, yolo_v3)

    # Step 6: Train/Evaluate the model
    trainer.train()
    # trainer.load("/home/yangxf/my_projects/machine_learning/checkpoints/auto_encoder/best_model.pth")
    # trainer.eval()


if __name__ == "__main__":
    main()
