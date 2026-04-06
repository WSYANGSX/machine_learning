from machine_learning.algorithms import MultimodalDetection
from machine_learning.evaluator import Evaluator


def main():
    # Step 1: Parse the data
    como = MultimodalDetection("como.yaml")

    # Step 2: Build the evaluate
    evaluator = Evaluator(
        como,
        "/home/yangxf/WorkSpace/machine_learning/checkpoints/como/vedai/best_model.pth",
        "vedai.yaml",
        False,
    )

    # Step 3: Evaluate the model
    evaluator.algorithm.eval(
        img_path="/home/yangxf/Downloads/vedai/3_co.png",
        ir_path="/home/yangxf/Downloads/vedai/3_ir.png",
        conf_thres=0.5,
        iou_thres=0.7,
        tag_size=0.35,
        modal="img",
    )


if __name__ == "__main__":
    main()
