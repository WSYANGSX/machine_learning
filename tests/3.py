import numpy as np

a = np.zeros((5, 4, 3))
b = np.zeros((0, 7, 8))
print(np.stack([a, b]))
