import torch

a = torch.rand((2, 5, 10))
print(a)
topk, topk_indices = torch.topk(a, 3, dim=-1, largest=True)
print(topk)
print(topk_indices)