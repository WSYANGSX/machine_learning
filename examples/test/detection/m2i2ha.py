from machine_learning.networks import M2I2HANet_v8
from machine_learning.algorithms import MultimodalDetection
from machine_learning.evaluator import Evaluator


def main():
    # Step 0: Build net
    net = M2I2HANet_v8(640, nc=5, net_scale="s")

    # Step 1: Parse the data
    como = MultimodalDetection("m2i2ha.yaml", net=net)

    # Step 2: Build the evaluate
    evaluator = Evaluator(
        como,
        "/home/yangxf/Downloads/old_train_data/checkpoints/m2i2ha/dv/m2i2ha-v8-s/best_model.pth",
        "drone_vehicle.yaml",
        False,
    )

    # Step 3: Evaluate the model
    # evaluator.eval("/home/yangxf/WorkSpace/machine_learning/data/coco-2017/images/test/000000580196.jpg")
    evaluator.eval(
        img_path="/home/yangxf/Downloads/dv/3_co.jpg",
        ir_path="/home/yangxf/Downloads/dv/3_ir.jpg",
        conf_thres=0.25,
        iou_thres=0.7,
        tag_size=0.35,
        modal="img",
    )


if __name__ == "__main__":
    main()
