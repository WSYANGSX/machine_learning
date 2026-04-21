from machine_learning.evaluator import Predictor
from machine_learning.algorithms.segmentation import PerPixelSegmentation


def main():
    # Step 1: Parse the data
    unet = PerPixelSegmentation("unet.yaml")

    # Step 2: Build the evaluate
    predictor = Predictor(
        unet,
        "/home/yangxf/WorkSpace/machine_learning/runs/unet/unet_car_2026-04-07_10-52/ckpt/best_model.pth",
        "car.yaml",
    )

    # Step 3: Predict
    predictor.algorithm.predict(stream="/home/yangxf/WorkSpace/datasets/..datasets/car/imgs/test/4b74275babf7_05.jpg")


if __name__ == "__main__":
    main()
