import torch

x = torch.randn(3, 4)  # shape [B, C, H]
y = torch.randn(3, 4)

print(torch.stack([x, y], dim=0))
print(torch.stack([x, y], dim=1))
print(torch.stack([x, y], dim=2))