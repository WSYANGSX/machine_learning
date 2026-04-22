from machine_learning.evaluator import Predictor


def main():
    # Step 1: Build the predictor from ckpt
    predictor = Predictor(
        "/home/yangxf/WorkSpace/machine_learning/runs/unet/unet__car_2026-04-22_11-29/ckpt/best_model.pth",
    )

    # Step 3: Predict
    predictor.algorithm.predict(stream="/home/yangxf/WorkSpace/datasets/..datasets/car/imgs/test/3f3e362dea23_01.jpg")


if __name__ == "__main__":
    main()
