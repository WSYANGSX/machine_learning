import torch

a = torch.tensor([[0.2, 0.5, 0.3, 0.4], [0.1, 0.2, 0.35, 0.4]])
b = a
b = b * 5
print(b, a)
