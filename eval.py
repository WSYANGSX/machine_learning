import argparse
from machine_learning.evaluator import Evaluator


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None, type=str, help="Restore from specified checkpoint")
    parser.add_argument("--device", type=str, default="auto", help="Running device")
    parser.add_argument("--plot", type=str, default=None, help="Whether to plot the results")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save the results")

    return parser


def main():
    # Step 1: Parse command line parameter
    opts = get_argparser().parse_args()

    # Step 2: Build the evaluate from ckpt
    evaluator = Evaluator(opts.ckpt, device=opts.device, plot=opts.plot, save_dir=opts.save_dir)

    # Step 3: Evaluate the model
    evaluator.eval()


if __name__ == "__main__":
    main()
