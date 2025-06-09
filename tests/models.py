import torch


bboxes = torch.randn((3, 2))
t = bboxes[:, None, ...]
print(t)
