from machine_learning.algorithms import MultimodalDetection
from machine_learning.evaluator import Evaluator


def main():
    # Step 1: Parse the data
    como = MultimodalDetection("como.yaml")

    # Step 2: Build the evaluate
    evaluator = Evaluator(
        como,
        "/home/yangxf/WorkSpace/machine_learning/checkpoints/como/como_vedai_2026-01-13_09-10/best_model.pth",
        "vedai.yaml",
        False,
    )

    # Step 3: Evaluate the model
    # evaluator.eval("/home/yangxf/WorkSpace/machine_learning/data/coco-2017/images/test/000000580196.jpg")
    evaluator.eval(
        img_path="/home/yangxf/Downloads/vedai/1040_co.png",
        ir_path="/home/yangxf/Downloads/vedai/1040_ir.png",
        conf_thres=0.25,
        iou_thres=0.7,
    )


if __name__ == "__main__":
    main()
