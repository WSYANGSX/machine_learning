import torch

a = torch.tensor([1, 2, 3], device="cuda")
print(id(a))
b = a.new(a.shape)
print(id(b))
