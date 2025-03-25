import math
import numpy as np
import torch

a = torch.randn((2, 3))
print(a)
a = a[None, ...]
print(a)