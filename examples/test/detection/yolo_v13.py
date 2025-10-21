from machine_learning.algorithms import YoloV13
from machine_learning.evaluator import Evaluator


def main():
    # Step 1: Build the algorithm
    yolov13 = YoloV13("yolo_v13.yaml")

    # Step 2: Build the evaluate
    evaluator = Evaluator(
        "/home/yangxf/WorkSpace/machine_learning/checkpoints/yolo_v13/2025-10-10_20-03/checkpoint_epoch_540.pth",
        yolov13,
        "coco-2017.yaml",
        False,
    )

    # Step 3: Evaluate the model
    # evaluator.eval("/home/yangxf/WorkSpace/machine_learning/data/coco-2017/images/test/000000580196.jpg")
    evaluator.eval(
        img_path="/home/yangxf/Downloads/3.jpg",
        conf_thres=0.4,
        iou_thres=0.7,
    )


if __name__ == "__main__":
    main()
