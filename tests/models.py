import numpy as np


a = [1, 2, 3]
print(np.array(a) % 10)
a = np.loadtxt("./1.txt").reshape(-1, 5)
print(a)
