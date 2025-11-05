import numpy as np
from itertools import chain

a = np.zeros((0, 5))
f = np.concatenate([a], axis=0)
print(f)
