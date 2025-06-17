from torchvision import transforms

from machine_learning.models.ae import Encoder, Decoder
from machine_learning.algorithms import AutoEncoder
from machine_learning.train import Trainer, TrainCfg
from machine_learning.utils.dataload import ParserFactory, ParserCfg


def main():
    # Step 1: Build the network
    input_size = (1, 28, 28)
    output_size = 128
    encoder = Encoder(input_size, output_size)
    decoder = Decoder(output_size)

    # Step 2: Build the algorithm
    auto_encoder = AutoEncoder(
        "./src/machine_learning/algorithms/generation/auto_encoder/config/config.yaml",
        {"encoder": encoder, "decoder": decoder},
    )

    # Step 3: Configure the augmentator/converter
    tfs = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081]),
        ]
    )

    # Step 4: Parse the data
    dataset_dir = "./data/minist"
    parser_cfg = ParserCfg(dataset_dir=dataset_dir, labels=True, transforms=tfs)
    parser = ParserFactory().parser_create(parser_cfg)
    dataset = parser.create()

    # Step 5: Configure the trainer
    trainer_cfg = TrainCfg(
        log_dir="./logs/auto_encoder/",
        model_dir="./checkpoints/auto_encoder/",
    )
    trainer = Trainer(trainer_cfg, dataset, auto_encoder)

    # Step 6: Train/Evaluate the model
    trainer.train()
    # trainer.load("/home/yangxf/my_projects/machine_learning/checkpoints/auto_encoder/best_model.pth")
    trainer.eval()


if __name__ == "__main__":
    main()
