import torch

targets = torch.tensor([[1, 2, 3, 4, 5], [4, 5, 6, 7, 8]])
b = torch.arange(5)
print(targets.matmul(b))
