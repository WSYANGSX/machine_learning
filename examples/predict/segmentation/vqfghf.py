from machine_learning.evaluator import Predictor
from machine_learning.algorithms.segmentation import PerPixelSegmentation


def main():
    # Step 1: Parse the data
    fghf_net = PerPixelSegmentation("vqfghf.yaml")

    # Step 2: Build the evaluate
    predictor = Predictor(
        fghf_net,
        "/home/yangxf/WorkSpace/machine_learning/runs/fghf/fghf_car_2026-04-13_18-34/ckpt/checkpoint_epoch_100.pth",
        "car.yaml",
    )

    # Step 3: Predict
    predictor.algorithm.predict(stream="/home/yangxf/WorkSpace/datasets/..datasets/car/imgs/train/0ce66b539f52_06.jpg")


if __name__ == "__main__":
    main()
