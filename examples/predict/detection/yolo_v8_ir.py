from machine_learning.algorithms.detection import YoloV8
from machine_learning.evaluator import Predictor


def main():
    # Step 1: Parse the data
    yolo_v8 = YoloV8("yolo_v8.yaml", amp=True, modality="ir")

    # Step 2: Configure the trainer
    predictor = Predictor(
        yolo_v8,
        "/home/yangxf/WorkSpace/machine_learning/runs/yolo_v8/yolo_v8_vedai_2026-04-18_15-36/ckpt/best_model.pth",
        "vedai.yaml",
    )

    # Step 3: Train the model
    predictor.algorithm.predict(
        stream="/home/yangxf/WorkSpace/datasets/..datasets/VEDAI/images/00000038_ir.png",
        conf_thres=0.25,
        iou_thres=0.7,
        tag_size=0.4,
        thickness=2,
    )


if __name__ == "__main__":
    main()
