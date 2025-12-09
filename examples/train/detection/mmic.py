from machine_learning.networks.yolo.mmic import MMICNet_with_v8_backbone
from machine_learning.algorithms.detection import MultimodalDetection
from machine_learning.trainer import Trainer, TrainerCfg


def main():
    # Step 0: build network
    net = MMICNet_with_v8_backbone(640, nc=5, net_scale="n")

    # Step 1: Parse the data
    mmic = MultimodalDetection("mmic.yaml", net=net, amp=True)

    # Step 2: Configure the trainer
    trainer_cfg = TrainerCfg(
        log_dir="/home/yangxf/WorkSpace/machine_learning/logs/mmic/",
        ckpt_dir="/home/yangxf/WorkSpace/machine_learning/checkpoints/mmic/",
        epochs=150,
        log_interval=10,
        save_interval=10,
        save_best=True,
    )
    trainer = Trainer(trainer_cfg, mmic, "drone_vehicle.yaml")

    # Step 3: Train the model
    trainer.train()


if __name__ == "__main__":
    main()
