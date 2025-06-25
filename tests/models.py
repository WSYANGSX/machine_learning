import torch

a = torch.tensor([[0.2, 0.5, 0.3, 0.4], [0.1, 0.2, 0.35, 0.4]])
b = torch.tensor([1, 2])
print(list(zip(a, b)))
