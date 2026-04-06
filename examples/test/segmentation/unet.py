from machine_learning.evaluator import Evaluator
from machine_learning.algorithms.segmentation import PerPixelSegmentation


def main():
    # Step 1: Parse the data
    unet = PerPixelSegmentation("unet.yaml")

    # Step 2: Build the evaluate
    evaluator = Evaluator(
        unet,
        "/home/yangxf/WorkSpace/machine_learning/runs/unet/unet_car_2026-04-06_12-00/ckpt/best_model.pth",
        "car.yaml",
        False,
    )

    # Step 3: Evaluate the model
    # evaluator.eval("/home/yangxf/WorkSpace/machine_learning/data/coco-2017/images/test/000000580196.jpg")
    evaluator.eval(img_path="/home/yangxf/WorkSpace/datasets/..datasets/car/imgs/train/0ed6904e1004_06.jpg")


if __name__ == "__main__":
    main()
