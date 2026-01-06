from machine_learning.algorithms.detection import YoloV8
from machine_learning.trainer import Trainer, TrainerCfg


def main():
    # Step 1: Parse the data
    yolo_v8 = YoloV8("yolo_v8.yaml", amp=True, modality="ir")

    # Step 2: Configure the trainer
    trainer_cfg = TrainerCfg(
        log_dir="/home/yangxf/WorkSpace/machine_learning/logs/yolo_v8/",
        ckpt_dir="/home/yangxf/WorkSpace/machine_learning/checkpoints/yolo_v8/",
        epochs=150,
        log_interval=10,
        save_interval=10,
        save_best=True,
    )
    trainer = Trainer(trainer_cfg, yolo_v8, "drone_vehicle.yaml")

    # Step 3: Train the model
    trainer.train()


if __name__ == "__main__":
    main()
