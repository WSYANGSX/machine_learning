from machine_learning.algorithms.detection import YoloV13
from machine_learning.trainer import Trainer, TrainerCfg


def main():
    # Step 1: Parse the data
    yolo_v13 = YoloV13("yolo_v13.yaml", amp=False)

    # Step 2: Configure the trainer
    trainer_cfg = TrainerCfg(
        log_dir="./logs/yolo_v13/",
        ckpt_dir="./checkpoints/yolo_v13/",
        epochs=600,
        log_interval=10,
        save_interval=10,
        save_best=True,
    )
    trainer = Trainer(trainer_cfg, yolo_v13, "coco-2017.yaml")

    # Step 3: Train the model
    trainer.train()


if __name__ == "__main__":
    main()
