import torch
import numpy as np

a = np.array([[1, 2, 4], [3, 7, 4], [5, 4, 9]])
b = a.clip(1, 2)
print(b)
