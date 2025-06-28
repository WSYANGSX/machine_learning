from torchvision import transforms
from machine_learning.algorithms import AutoEncoder
from machine_learning.models.ae import Encoder, Decoder
from machine_learning.utils.dataload import ParserFactory, ParserCfg


def main():
    # Step 1: Configure the augmentator/converter and parse the data
    tfs = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.1307], std=[0.3081])])

    dataset_dir = "./data/minist"
    parser_cfg = ParserCfg(dataset_dir=dataset_dir, labels=True, tfs=tfs)
    parser = ParserFactory().create_parser(parser_cfg)
    data = parser.create()

    # Step 2: Build the network
    input_size = (1, 28, 28)
    output_size = 128
    encoder = Encoder(input_size, output_size)
    decoder = Decoder(output_size)

    # Step 3: Build the algorithm
    auto_encoder = AutoEncoder(
        "./src/machine_learning/algorithms/generation/auto_encoder/config/config.yaml",
        {"encoder": encoder, "decoder": decoder},
        data=data,
    )

    # Step 6: Evaluate the model
    auto_encoder.load("/home/yangxf/WorkSpace/machine_learning/checkpoints/auto_encoder/best_model.pth")
    auto_encoder.eval(5)


if __name__ == "__main__":
    main()
