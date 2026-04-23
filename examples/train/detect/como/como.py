import time
from machine_learning.trainer import Trainer, TrainerCfg


def main():
    # step 1: setup trainer
    trainer_cfg = TrainerCfg(
        epochs=100,
        log_interval=10,
        save_interval=10,
        save_best=True,
        seed=int(time.time()),
        amp=True,
    )
    trainer = Trainer("como", trainer_cfg, "como.yaml", "vedai.yaml")

    # Step 2: train
    trainer.train()


if __name__ == "__main__":
    main()
