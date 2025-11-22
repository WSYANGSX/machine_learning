import torch
import numpy as np


target = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
print(target)
patch = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
print(patch)
target[:, :] = patch[..., None]
print(target)

# target = torch.randint(0, 255, (3, 32, 32), dtype=torch.uint8)
# patch = torch.randint(0, 255, (32, 32), dtype=torch.uint8)
# # b = np.repeat(target[:, :, None], 3, axis=2)
# # print(b)
# target[:, :] = patch
