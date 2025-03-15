import torch.nn as nn
from machine_learning.models import CNN
from collections import OrderedDict

input_size = (3, 256, 256)
output_size = (20, 4, 4)
hidden_layers = OrderedDict(
    [
        ("conv1", nn.Conv2d(3, 10, 4, 2, 0)),  # (10, 127, 127)
        ("pooling1", nn.AvgPool2d(4, 2, 0)),  # (10, 62, 62)
        ("batchnorm1", nn.BatchNorm2d(10)),
        ("relu1", nn.ReLU()),
        ("conv2", nn.Conv2d(10, 5, 3, 2, 1)),  # (5, 31, 31)
        ("pooling2", nn.AvgPool2d(2, 2, 0)),  # (5,15,15)   向下取整
        ("batchnorm2", nn.BatchNorm2d(5)),
        ("relu2", nn.ReLU()),
        ("conv3", nn.Conv2d(5, 20, 2, 2, 0)),  # (20, 7, 7)
        ("pooling3", nn.AvgPool2d(2, 2, 1, ceil_mode=True)),  # (20,5,5)
        ("batchnorm3", nn.BatchNorm2d(20)),
        ("relu3", nn.ReLU()),
    ]
)
a = CNN(input_size=input_size, output_size=output_size, hidden_layers=hidden_layers)
a.view_structure()
