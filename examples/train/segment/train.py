import argparse
from machine_learning.trainer import Trainer, TrainerCfg


def get_argparser():
    parser = argparse.ArgumentParser()

    # Segmentation Algorithm Options
    parser.add_argument("--name", type=str, default="fghf", help="Algorithm name")
    parser.add_argument("--algo_cfg", type=str, default="fghf.yaml", help="Algorithm configuration yaml file path")
    parser.add_argument("--net_scale", type=str, default="n", help="The size of the network (default: n)")

    # Train Options
    parser.add_argument("--seed", type=int, default=23, help="Global random seed (default: 23)")
    parser.add_argument("--epochs", type=int, default=100, help="Train epoch number (default: 100)")
    parser.add_argument("--dataset", type=str, default="car.yaml", help="Configure of dataset")
    parser.add_argument("--log_interval", type=int, default=10, help="How many batches the data is recorded every time")
    parser.add_argument("--save_interval", type=int, default=10, help="How many epochs to record data")
    parser.add_argument("--save_best", type=bool, default=True, help="Whether save the best model during training")
    parser.add_argument("--amp", type=bool, default=False, help="Whether to enable Automatic Mixed Precision")
    parser.add_argument("--ema", type=bool, default=False, help="Whether to enable Exponential Moving Average")
    parser.add_argument("--device", type=str, default="auto", help="Running device")
    parser.add_argument("--continue_training", action="store_true", default=False)
    parser.add_argument("--resume", default=None, type=str, help="restore from lastest checkpoint in resume directory")
    parser.add_argument("--ckpt", default=None, type=str, help="restore from specified checkpoint")

    # Train Options
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate (default: 0.01)")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size (default: 16)")
    parser.add_argument(
        "--lr_policy", type=str, default="poly", choices=["poly", "step"], help="learning rate scheduler policy"
    )
    parser.add_argument("--augment", type=bool, default=True, help="Whether augment data during the training process")

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

    # Step 3: Overwrite the algorithm parameters (Optional)

    # Step 4: Instantiate the trainer
    trainer = Trainer(
        name=opts.name,
        train_cfg=trainer_cfg,
        algorithm_cfg=opts.algo_cfg,
        dataset_cfg=opts.dataset,
    )

    # Step 3: Train the model
    trainer.train()


if __name__ == "__main__":
    main()
