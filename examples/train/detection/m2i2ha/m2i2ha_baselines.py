import time
import argparse
from machine_learning.networks.yolo.m2i2ha import SimpleFusion_v8
from machine_learning.algorithms.detection import MultimodalDetection
from machine_learning.trainer import Trainer, TrainerCfg


def main(opt):
    # Step 0: build network (optional)
    net = SimpleFusion_v8(640, nc=5, net_scale="s", hyperace=opt.hyperace)

    # Step 1: Parse the data
    m2i2ha = MultimodalDetection("m2i2ha.yaml", net=net, amp=False)

    # Step 2: Configure the trainer
    trainer_cfg = TrainerCfg(
        epochs=150,
        log_interval=10,
        save_interval=10,
        save_best=True,
        seed=int(time.time()),
    )
    trainer = Trainer(trainer_cfg, m2i2ha, "drone_vehicle.yaml")

    # Step 3: Train the model
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyperace", action="store_true", default=False)
    opt = parser.parse_args()

    main(opt)
