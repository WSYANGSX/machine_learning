from machine_learning.evaluator import Predictor


def main():
    # Step 1: Build the predictor from ckpt
    predictor = Predictor(
        "/home/yangxf/WorkSpace/machine_learning/runs/fghf/fghf_s_sbd_2026-04-25_14-05/ckpt/best_model.pth",
    )

    # Step 3: Predict
    predictor.algorithm.predict(
        stream="/home/yangxf/WorkSpace/datasets/..datasets/VOC2012/JPEGImages/2007_000061.jpg",
    )


if __name__ == "__main__":
    main()
