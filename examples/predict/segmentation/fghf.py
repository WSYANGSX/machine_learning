from machine_learning.evaluator import Predictor
from machine_learning.algorithms.segmentation import PerPixelSegmentation


def main():
    # Step 1: Parse the data
    fghf_net = PerPixelSegmentation("fghf.yaml")

    # Step 2: Build the evaluate
    predictor = Predictor(
        fghf_net,
        "/home/yangxf/WorkSpace/machine_learning/runs/fghf/fghf_car_2026-04-20_19-14/ckpt/best_model.pth",
        "car.yaml",
    )

    # Step 3: Predict
    predictor.algorithm.predict(stream="/home/yangxf/WorkSpace/datasets/..datasets/car/imgs/test/3f3e362dea23_01.jpg")


if __name__ == "__main__":
    main()
