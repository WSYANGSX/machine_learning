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
        "/home/yangxf/WorkSpace/machine_learning/checkpoints/m2i2ha/vedai/s/best_model.pth",
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
