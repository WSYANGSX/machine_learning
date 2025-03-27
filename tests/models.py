import torch
from machine_learning.models.unet import TimestepEmbedding

a = TimestepEmbedding(100)
t = torch.randint(1, 100, (15,))
print(a.forward(t))
