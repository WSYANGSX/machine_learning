import torch


a = torch.tensor([1, 2, 3], device="cuda", dtype=torch.int32)
b = torch.tensor([4, 5, 6], device="cuda", dtype=torch.float32)
print(torch.cat([a, b], dim=0).dtype)
