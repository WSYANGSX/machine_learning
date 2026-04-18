from machine_learning.evaluator import Predictor
from machine_learning.algorithms.segmentation import PerPixelSegmentation


def main():
    # Step 1: Parse the data
    unet = PerPixelSegmentation("unet.yaml")

    # Step 2: Build the evaluate
    predictor = Predictor(
        unet,
        "/home/yangxf/WorkSpace/machine_learning/runs/unet/unet_sbd_2026-04-15_16-43/ckpt/best_model.pth",
        "sbd.yaml",
    )

    # Step 3: Predict
    predictor.algorithm.predict(stream="/home/yangxf/WorkSpace/datasets/..datasets/VOC2012/JPEGImages/2007_000027.jpg")


if __name__ == "__main__":
    main()
