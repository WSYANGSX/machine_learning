import time
from machine_learning.algorithms.detection import MultimodalDetection
from machine_learning.trainer import Trainer, TrainerCfg


def main():
    # Step 1: Parse the data
    como = MultimodalDetection("como.yaml", amp=False)

    # Step 2: Configure the trainer
    trainer_cfg = TrainerCfg(
        log_dir="/home/yangxf/WorkSpace/machine_learning/logs/como/",
        ckpt_dir="/home/yangxf/WorkSpace/machine_learning/checkpoints/como/",
        epochs=60,
        log_interval=10,
        save_interval=10,
        save_best=True,
        seed=int(time.time()),
    )
    trainer = Trainer(trainer_cfg, como, "llvip.yaml")

    # Step 3: Train the model
    trainer.train()


if __name__ == "__main__":
    main()
