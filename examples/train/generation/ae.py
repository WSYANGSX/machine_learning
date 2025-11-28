from machine_learning.algorithms import AutoEncoder
from machine_learning.trainer import Trainer, TrainerCfg


def main():
    # Step 1: Build the algorithm
    auto_encoder = AutoEncoder("ae.yaml")

    # Step 2: Configure the trainer
    trainer_cfg = TrainerCfg(
        log_dir="./logs/auto_encoder/",
        ckpt_dir="./checkpoints/auto_encoder/",
    )
    trainer = Trainer(trainer_cfg, auto_encoder, "minist.yaml")

    # Step 3: Train the model
    trainer.train()
    # trainer.train_from_checkpoint(
    #     "/home/yangxf/WorkSpace/machine_learning/checkpoints/auto_encoder/2025-10-14_21-58/checkpoint_epoch_20.pth"
    # )


if __name__ == "__main__":
    main()
