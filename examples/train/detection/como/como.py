import time
from machine_learning.trainer import Trainer, TrainerCfg
from machine_learning.algorithms.detection import MultimodalDetection


def main():
    # Step 1: Parse the data
    como = MultimodalDetection("como.yaml", amp=False)

    # Step 2: Configure the trainer
    trainer_cfg = TrainerCfg(
        epochs=300,
        log_interval=10,
        save_interval=10,
        save_best=True,
        seed=int(time.time()),
    )
    trainer = Trainer(trainer_cfg, como, "vedai.yaml")

    # Step 3: Train the model
    trainer.train()


if __name__ == "__main__":
    main()
