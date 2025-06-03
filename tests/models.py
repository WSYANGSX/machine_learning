import numpy as np

a = np.arange(95, 99).reshape(2, 2, 1)
print(a)
b = np.pad(a, ((1, 2), (2, 3), (2, 3)), "constant", constant_values=0)
print(b)
