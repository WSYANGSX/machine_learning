import argparse
from machine_learning.evaluator import Predictor


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None, type=str, help="Restore from specified checkpoint")
    parser.add_argument("--device", type=str, default="auto", help="Running device")
    parser.add_argument("--stream", type=str, default=None, help="The file stream to be predicted")

    return parser


def main():
    # Step 1: Parse command line parameter
    opts = get_argparser().parse_args()

    # Step 2: Build the predictor from ckpt
    predictor = Predictor(opts.ckpt, device=opts.device)

    # Step 3: Predict
    predictor.algorithm.predict(stream=opts.stream)


if __name__ == "__main__":
    main()
