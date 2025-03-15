import torch
from machine_learning.algorithms import AlgorithmBase

optimizer = torch.optim.Adam(
    params=self.parameters(),
    lr=0.11,
    betas=(0.9, 0.99),
    eps=opt_config["eps"],
    weight_decay=opt_config["weight_decay"],
)
