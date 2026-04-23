import time
from machine_learning.trainer import Trainer, TrainerCfg


def main():
    # Step 1: Setup the trainer
    trainer_cfg = TrainerCfg(
        epochs=300,
        log_interval=10,
        save_interval=10,
        save_best=True,
        seed=int(time.time()),
    )
    trainer = Trainer("yolo_v8", trainer_cfg, "yolo_v8.yaml", "vedai.yaml")

    # Step 2: Train the model
    trainer.train()


if __name__ == "__main__":
    main()
