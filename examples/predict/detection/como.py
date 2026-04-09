from machine_learning.evaluator import Predictor
from machine_learning.algorithms import MultimodalDetection


def main():
    # Step 1: Parse the data
    como = MultimodalDetection("como.yaml")

    # Step 2: Build the evaluate
    predictor = Predictor(
        como,
        "/home/yangxf/WorkSpace/machine_learning/runs/como/como_vedai_2026-04-06_17-05/ckpt/best_model.pth",
        "vedai.yaml",
    )

    # Step 3: Evaluate the model
    predictor.algorithm.predict(
        stream={
            "img": "/home/yangxf/WorkSpace/datasets/..datasets/VEDAI/images/00000027_co.png",
            "ir": "/home/yangxf/WorkSpace/datasets/..datasets/VEDAI/images/00000027_ir.png",
        },
        conf_thres=0.5,
        iou_thres=0.7,
        tag_size=0.35,
        base="img",
    )


if __name__ == "__main__":
    main()
