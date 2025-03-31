import torch
import tqdm

a = torch.linspace(0.0001, 0.002, 2000)
print(a.shape)

for i in tqdm.trange(10, -1, -1):
    print(i)
