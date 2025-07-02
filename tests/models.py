import torch


anchor_ids = torch.randn((3, 3, 2))
indices = (torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([0, 1]))
print(anchor_ids)
print(anchor_ids[indices])
