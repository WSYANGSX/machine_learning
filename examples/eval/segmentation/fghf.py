from machine_learning.evaluator import Evaluator
from machine_learning.algorithms.segmentation import PerPixelSegmentation


def main():
    # Step 1: Parse the data
    fghf_net = PerPixelSegmentation("fghf.yaml")

    # Step 2: Build the evaluate
    evaluator = Evaluator(
        fghf_net,
        "/home/yangxf/WorkSpace/machine_learning/runs/fghf/fghf_car_2026-04-21_09-50/ckpt/best_model.pth",
        "car.yaml",
        True,
    )

    # Step 3: Evaluate the model
    evaluator.eval()


if __name__ == "__main__":
    main()
