import torch
import numpy as np

a = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(a.shape)
b = np.array(a)
print(type(b.shape))
