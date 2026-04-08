from machine_learning.algorithms.detection import YoloV8
from machine_learning.evaluator import Evaluator


def main():
    # Step 1: Parse the data
    yolo_v8 = YoloV8("yolo_v8.yaml", amp=True, modality="ir")

    # Step 2: Configure the trainer
    evaluator = Evaluator(
        yolo_v8,
        "/home/yangxf/WorkSpace/machine_learning/runs/yolo_v8/yolo_v8_vedai_2026-04-08_16-05/ckpt/best_model.pth",
        "vedai.yaml",
        True,
    )

    # Step 3: Train the model
    evaluator.eval()


if __name__ == "__main__":
    main()
