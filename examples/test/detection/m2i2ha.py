from machine_learning.networks import M2I2HANet_v8
from machine_learning.algorithms import MultimodalDetection
from machine_learning.evaluator import Evaluator


def main():
    # Step 0: Build net
    net = M2I2HANet_v8(640, nc=8, net_scale="s")

    # Step 1: Parse the data
    como = MultimodalDetection("m2i2ha.yaml", net=net)

    # Step 2: Build the evaluate
    evaluator = Evaluator(
        como,
        "/home/yangxf/WorkSpace/machine_learning/runs/m2i2ha/m2i2ha_vedai_2026-01-17_18-09/ckpt/best_model.pth",
        "vedai.yaml",
        False,
    )

    # Step 3: Evaluate the model
    # evaluator.eval("/home/yangxf/WorkSpace/machine_learning/data/coco-2017/images/test/000000580196.jpg")
    evaluator.eval(
        img_path="/home/yangxf/Downloads/vedai/3_co.png",
        ir_path="/home/yangxf/Downloads/vedai/3_ir.png",
        conf_thres=0.25,
        iou_thres=0.7,
        tag_size=0.35,
        modal="img",
    )


if __name__ == "__main__":
    main()
