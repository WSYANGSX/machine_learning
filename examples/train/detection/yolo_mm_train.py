from machine_learning.networks.yolo import NblityNet
from machine_learning.algorithms.detection import YoloMM
from machine_learning.train import Trainer, TrainCfg
from machine_learning.utils.aug import DEFAULT_YOLOMM_AUG
from machine_learning.utils.transforms import ImgTransform
from machine_learning.data.parsers import YoloMMParser, YoloParserCfg
from machine_learning.utils import load_config_from_yaml


def main():
    # Step 1: Parse the data
    tfs = ImgTransform(
        aug_cfg=DEFAULT_YOLOMM_AUG,
        normalize=True,
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_tensor=True,
    )

    parser_cfg = YoloParserCfg(dataset_dir="./data/Flir_aligned", labels=True, tfs=tfs, multiscale=False)
    data = YoloMMParser(parser_cfg).create()  # (class_names, train_dataset, val_dataset)

    # Step 1: Parse configurations
    yolomm_cfg = load_config_from_yaml("./src/machine_learning/algorithms/detection/yolo_mm/config/yolo_mm.yaml")

    # Step 2: Build networks
    num_classes = data["class_nums"]
    img_size = yolomm_cfg["algorithm"]["img_size"]
    nblitynet = NblityNet(img_shape=(3, img_size, img_size), thermal_shape=(1, img_size, img_size), nc=num_classes)

    # Step 2: Build the algorithm
    yolo_mm = YoloMM(cfg=yolomm_cfg, net=nblitynet)

    # Step 5: Configure the trainer
    trainer_cfg = TrainCfg(
        log_dir="./logs/yolomm/",
        model_dir="./checkpoints/yolomm/",
        epochs=600,
        log_interval=10,
        save_interval=10,
        save_best=True,
    )
    trainer = Trainer(trainer_cfg, yolo_mm, data)

    # Step 6: Train the model
    trainer.train()
    # trainer.train_from_checkpoint("/home/yangxf/WorkSpace/machine_learning/checkpoints/yolov3/checkpoint_epoch_19.pth")


if __name__ == "__main__":
    main()
