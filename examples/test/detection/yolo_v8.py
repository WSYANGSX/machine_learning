from machine_learning.algorithms import YoloV8
from machine_learning.evaluator import Evaluator


def main():
    # Step 1: Build the algorithm
    yolov8 = YoloV8("yolo_v8.yaml")

    # Step 2: Build the evaluate
    evaluator = Evaluator(
        yolov8,
        "/home/yangxf/WorkSpace/machine_learning/checkpoints/yolo_v8/vedai/rgb/checkpoint_epoch_300.pth",
        "vedai.yaml",
        False,
    )

    # Step 3: Evaluate the model
    # evaluator.eval("/home/yangxf/WorkSpace/machine_learning/data/coco-2017/images/test/000000580196.jpg")
    evaluator.eval(img_path="/home/yangxf/Downloads/vedai/3_co.png", conf_thres=0.5, iou_thres=0.7, tag_size=0.35)


if __name__ == "__main__":
    main()
