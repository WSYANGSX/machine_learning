from machine_learning.evaluator import Evaluator
from machine_learning.algorithms.segmentation import PerPixelSegmentation


def main():
    # Step 1: Parse the data
    unet = PerPixelSegmentation("unet.yaml")

    # Step 2: Build the evaluate
    evaluator = Evaluator(
        unet,
        "/home/yangxf/WorkSpace/machine_learning/runs/unet/unet_sbd_2026-04-15_16-43/ckpt/best_model.pth",
        "sbd.yaml",
        True,
    )

    # Step 3: Evaluate the model
    evaluator.eval()


if __name__ == "__main__":
    main()
