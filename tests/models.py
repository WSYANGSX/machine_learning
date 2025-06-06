import torch
from torchvision.transforms import ToTensor

a = torch.randn((2, 3))
print(a)
print(a[:, :, None])
