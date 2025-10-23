from numbers import Number, Real, Complex
import numpy as np
import torch

a = torch.scalar_tensor(1, dtype=torch.float32)
print(isinstance(a, Number))
