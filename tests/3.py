import time
import torch
import numpy as np

_targets = ("img", "ir", "depth")
sample = {"img": np.random.randn(32, 32, 3), "ir": np.random.randn(32, 32, 3), "depth": np.random.randn(32, 32, 3)}


t0 = time.time()
size = next(
    iter([sample[t].shape[:2] for t in _targets if t in sample and sample[t] is not None]),
    None,
)
t1 = time.time()
for t in _targets:
    if t in sample:
        size = sample[t].shape[:2]
        break
t2 = time.time()

print(t1 - t0, t2 - t1)
