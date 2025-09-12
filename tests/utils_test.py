import numpy as np
from pympler import asizeof
from machine_learning.utils.detection import class_maps

b = ["cat", "pig", "dog"]
c = class_maps(b)
print(c["0"])
