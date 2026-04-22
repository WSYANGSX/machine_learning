import time
from machine_learning.trainer import Trainer, TrainerCfg


def main():
    # step 1: set trainer cfg
    trainer_cfg = TrainerCfg(
        epochs=100,
        log_interval=10,
        save_interval=10,
        save_best=True,
        seed=int(time.time()),
        amp=True,
        continue_training=True,
        resume="/home/yangxf/WorkSpace/machine_learning/runs/unet/unet__car_2026-04-22_11-29",
    )
    trainer = Trainer("unet", trainer_cfg, "unet.yaml", "car.yaml")

    # Step 3: Train the model
    trainer.train()


if __name__ == "__main__":
    main()
