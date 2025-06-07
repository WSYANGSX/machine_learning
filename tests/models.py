import torch

a = torch.randn((2, 3))
print(a)
print(a[:, :, None])
