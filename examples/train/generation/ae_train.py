from machine_learning.algorithms import AutoEncoder
from machine_learning.train import Trainer, TrainCfg
from machine_learning.networks.auto_encoders import AENet
from machine_learning.utils.transforms import ImgTransform
from machine_learning.utils.transforms import DEFAULT_AUG
from machine_learning.data.parsers import ParserCfg, MinistParser


def main():
    # Step 1: Configure the augmentator/converter and parse the data
    tfs = ImgTransform(aug_cfg=DEFAULT_AUG, normalize=True, mean=[0.1307], std=[0.3081], to_tensor=True)

    dataset_dir = "./data/minist"
    parser_cfg = ParserCfg(dataset_dir=dataset_dir, labels=True, tfs=tfs)
    parser = MinistParser(parser_cfg)
    data = parser.create()

    # Step 2: Build the network
    image_shape = data["image_shape"]
    output_size = 128
    net = AENet(image_shape, output_size)

    # Step 3: Build the algorithm
    auto_encoder = AutoEncoder(
        "./src/machine_learning/algorithms/generation/auto_encoder/config/config.yaml",
        net,
    )

    # Step 4: Configure the trainer
    trainer_cfg = TrainCfg(
        log_dir="./logs/auto_encoder/",
        model_dir="./checkpoints/auto_encoder/",
    )
    trainer = Trainer(trainer_cfg, auto_encoder, data)

    # Step 5: Train the model
    trainer.train()

    # Step 6: Evaluate the model
    # auto_encoder.load("/home/yangxf/my_projects/machine_learning/checkpoints/auto_encoder/best_model.pth")
    auto_encoder.eval(5)


if __name__ == "__main__":
    main()
