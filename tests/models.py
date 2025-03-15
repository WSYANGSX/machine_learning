import torch.nn as nn
from machine_learning.models import MLP
from collections import OrderedDict

hidden_layers = OrderedDict(
    {"lin1": nn.Linear(10, 20), "relu1": nn.ReLU(), "lin2": nn.Linear(20, 5), "relu2": nn.ReLU()}
)
a = MLP(input_dim=10, output_dim=5, hidden_layers=hidden_layers)
a.view_structure()
