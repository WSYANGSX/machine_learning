from machine_learning.algorithms import AutoEncoder
from machine_learning.evaluator import Evaluator


def main():
    # Step 1: Build the algorithm
    auto_encoder = AutoEncoder("ae.yaml")

    # Step 2: Build the evaluate
    evaluator = Evaluator(
        "/home/yangxf/WorkSpace/machine_learning/checkpoints/auto_encoder/2025-10-14_21-58/best_model.pth",
        "minist.yaml",
        auto_encoder,
    )

    # Step 3: Evaluate the model
    evaluator.eval(5)


if __name__ == "__main__":
    main()
