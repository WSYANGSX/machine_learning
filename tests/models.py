import numpy as np
import torch
from torchvision.transforms import ToTensor

a = np.array([], dtype=np.uint8)
b = torch.from_numpy(a)
print(b)
