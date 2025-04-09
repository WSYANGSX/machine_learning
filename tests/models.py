import torch

a = torch.tensor([[1, 2], [5, 5]])
b = torch.tensor([[10, 0], [5, 8]])

a = torch.unsqueeze(a, dim=1)
print(a - b)
print((a - b) ** 2)
c = torch.sqrt(torch.sum((a - b) ** 2, dim=-1))
index = torch.argmin(c, dim=-1)
print(index)
print(b[index])
