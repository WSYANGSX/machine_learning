import torch

a = {"1": 1, "2": 2}
b = {"1": 3, "4": 4}
a.update(b)
print(iter(a.values())[0])
