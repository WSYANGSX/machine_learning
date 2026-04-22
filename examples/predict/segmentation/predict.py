from machine_learning.evaluator import Predictor


def main():
    # Step 1: Build the predictor from ckpt
    predictor = Predictor(
        "/home/yangxf/WorkSpace/machine_learning/runs/unet/unet_sbd_2026-04-15_16-43/ckpt/best_model.pth",
    )

    # Step 3: Predict
    predictor.algorithm.predict(stream="/home/yangxf/WorkSpace/datasets/..datasets/car/imgs/test/3f202616a2b9_11.jpg")


if __name__ == "__main__":
    main()
