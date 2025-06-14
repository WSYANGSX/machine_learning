import torch

a = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [3, 2, 5, 4], [6, 7, 5, 1]])
print(a[[1, 3], [3, 2]])
