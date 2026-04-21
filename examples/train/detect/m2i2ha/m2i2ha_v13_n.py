import time
from machine_learning.networks.yolo.m2i2ha import M2I2HANet_v13
from machine_learning.algorithms.detection import MultimodalDetection
from machine_learning.trainer import Trainer, TrainerCfg


def main():
    # Step 0: build network (optional)
    net = M2I2HANet_v13(640, nc=1, net_scale="n")

    # Step 1: Parse the data
    m2i2ha = MultimodalDetection("m2i2ha.yaml", net=net, amp=False)

    # Step 2: Configure the trainer
    trainer_cfg = TrainerCfg(
        log_dir="/home/yangxf/WorkSpace/machine_learning/logs/m2i2ha/",
        ckpt_dir="/home/yangxf/WorkSpace/machine_learning/checkpoints/m2i2ha/",
        epochs=60,
        log_interval=10,
        save_interval=10,
        save_best=True,
        seed=int(time.time()),
    )
    trainer = Trainer(trainer_cfg, m2i2ha, "llvip.yaml")

    # Step 3: Train the model
    trainer.train()


if __name__ == "__main__":
    main()
