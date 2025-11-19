import numpy as np

masks = np.zeros((1, 4, 5))
b = np.zeros((0, masks.shape[1], masks.shape[2]), dtype=masks.dtype)
print(b)
