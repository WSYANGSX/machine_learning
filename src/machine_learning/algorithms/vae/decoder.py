from collections import OrderedDict

import torch.nn as nn
from torchinfo import summary


class Decoder(nn.Module):
    def __init__(self, input_size: tuple[int]) -> None:
        super().__init__()

        self.input_size = input_size

        self.layer1 = nn.Sequential(
            OrderedDict(
                [
                    ("linear", nn.Linear(input_size[-1], 294)),
                    ("reshape", nn.Unflatten(1, (6, 7, 7))),
                ]
            ),
        )

        self.layer2 = nn.Sequential(
            OrderedDict(
                [
                    ("BatchNorm1", nn.BatchNorm2d(6)),
                    ("relu1", nn.ReLU()),
                    (
                        "deconv1",
                        nn.ConvTranspose2d(6, 3, 3, 2, 1, output_padding=1),
                    ),
                ]
            ),
        )  # 输出尺寸：(16,12,12))
        self.layer3 = nn.Sequential(
            OrderedDict(
                [
                    ("BatchNorm2", nn.BatchNorm2d(3)),
                    ("relu2", nn.ReLU()),
                    (
                        "deconv2",
                        nn.ConvTranspose2d(3, 1, 3, 2, 1, output_padding=1),
                    ),
                ]
            )
        )  # 输出尺寸：(16,12,12))
        self.layer4 = nn.Sigmoid()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        output1 = self.layer1(x)
        output2 = self.layer2(output1)
        output3 = self.layer3(output2)
        output4 = self.layer4(output3)

        return output4

    def view_structure(self):
        summary(self, input_size=self.input_size)

    def view_modules(self):
        for module in self.named_children():
            print(module)


if __name__ == "__main__":
    decoder = Decoder((1, 128))
    decoder._initialize_weights()
    decoder.view_structure()
