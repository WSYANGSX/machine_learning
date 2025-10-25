import torch
from torch.utils.data._utils.collate import default_collate
import numpy as np

a = [1, 2, 3, 4, 5, 6]
b = torch.cat(a, dim=0)
print(b)
