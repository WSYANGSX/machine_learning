import argparse

from machine_learning.trainer import Trainer, TrainerCfg
from machine_learning.algorithms import get_alogrithm_cfg


def get_argparser():
    parser = argparse.ArgumentParser()

    # Main Segmentation Algorithm Options
    parser.add_argument("--name", type=str, default="fghf", help="Algorithm name")
    parser.add_argument("--cfg", type=str, default="fghf.yaml", help="Algorithm configuration yaml file path")
    parser.add_argument("--net_scale", type=str, default=None, help="The size of the network (default: n)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: 0.01)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (default: 16)")
    parser.set_defaults(augment=None)
    parser.add_argument("--augment", dest="augment", action="store_true", help="Enable data augment during training")
    parser.add_argument(
        "--no-augment", dest="augment", action="store_false", help="Disable data augment during training"
    )
    parser.add_argument("--imgsz", type=int, default=None, help="Image size (default: 640)")
    parser.add_argument("--close_mosaic_epoch", type=float, default=None, help="The proportion of data enhanced epochs")

    # Train Options
    parser.add_argument("--seed", type=int, default=23, help="Global random seed (default: 23)")
    parser.add_argument("--epochs", type=int, default=100, help="Train epoch number (default: 100)")
    parser.add_argument("--dataset", type=str, default="car.yaml", help="Configure of dataset")
    parser.add_argument("--log_interval", type=int, default=10, help="How many batches the data is recorded every time")
    parser.add_argument("--save_interval", type=int, default=10, help="How many epochs to record data")
    parser.set_defaults(save_best=True)
    parser.add_argument("--save_best", dest="save_best", action="store_true", help="Save the best model")
    parser.add_argument("--no-save_best", dest="save_best", action="store_false", help="Do not save the best model")
    parser.add_argument("--device", type=str, default="auto", help="Running device")
    parser.add_argument("--amp", action="store_true", help="Whether to enable Automatic Mixed Precision")
    parser.add_argument("--ema", action="store_true", help="Whether to enable Exponential Moving Average")
    parser.add_argument("--continue_training", action="store_true", help="Whether to continue training from checkpoint")
    parser.add_argument("--resume", default=None, type=str, help="Restore from lastest checkpoint in resume directory")
    parser.add_argument("--ckpt", default=None, type=str, help="Restore from specified checkpoint")

    return parser


def main():
    # Step 1: Parse command line parameter
    opts = get_argparser().parse_args()

    # Step 2: Set training parameters
    trainer_cfg = TrainerCfg(
        seed=opts.seed,
        epochs=opts.epochs,
        log_interval=opts.log_interval,
        save_interval=opts.save_interval,
        save_best=opts.save_best,
        amp=opts.amp,
        ema=opts.ema,
        device=opts.device,
        continue_training=opts.continue_training,
        resume=opts.resume,
        ckpt=opts.ckpt,
    )

    # Step 3: Set algorithm parameters
    algo_cfg = get_alogrithm_cfg(
        opts.cfg,
        {
            k: v
            for k, v in vars(opts).items()
            if v is not None and k not in list(TrainerCfg.__annotations__.keys()) + ["cfg", "name", "dataset"]
        },
    )

    # Step 4: Instantiate the trainer
    trainer = Trainer(
        name=opts.name,
        train_cfg=trainer_cfg,
        algorithm_cfg=algo_cfg,
        dataset_cfg=opts.dataset,
    )

    # Step 3: Train the model
    trainer.train()


if __name__ == "__main__":
    main()
