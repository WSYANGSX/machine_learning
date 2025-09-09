from machine_learning.algorithms import AutoEncoder
from machine_learning.trainer import Trainer, TrainCfg


def main():
    # Step 1: Build the algorithm
    auto_encoder = AutoEncoder("ae.yaml")

    # Step 2: Configure the trainer
    trainer_cfg = TrainCfg(
        log_dir="./logs/auto_encoder/",
        ckpt_dir="./checkpoints/auto_encoder/",
    )
    trainer = Trainer(trainer_cfg, "minist.yaml", auto_encoder)

    # Step 3: Train the model
    trainer.train()

    # Step 4: Evaluate the model
    # auto_encoder.load(
    #     "/home/yangxf/WorkSpace/machine_learning/checkpoints/auto_encoder/2025-09-06_11-34/best_model.pth"
    # )
    auto_encoder.eval(5)


if __name__ == "__main__":
    main()
