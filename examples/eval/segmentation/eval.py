from machine_learning.evaluator import Evaluator


def main():
    # Step 1: Build the evaluate from ckpt
    evaluator = Evaluator(
        "/home/yangxf/WorkSpace/machine_learning/runs/unet/unet_sbd_2026-04-15_16-43/ckpt/best_model.pth",
        device="cuda:1",
    )

    # Step 3: Evaluate the model
    evaluator.eval()


if __name__ == "__main__":
    main()
