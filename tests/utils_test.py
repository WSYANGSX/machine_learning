import torch
import random

a = torch.arange(3).repeat(5, 1).T.view(3, 5, 1)
print(a)