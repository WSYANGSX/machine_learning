import torch


a = torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
b = torch.tensor([[5.0, 6.0, 3.0, 4.0], [5.0, 6.0, 3.0, 4.0]])

c = torch.stack((a, b), dim=1)
print(c) 