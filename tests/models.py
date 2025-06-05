import torch
import numpy as np

a = torch.tensor(np.array([])).reshape(-1, 3)
b = torch.tensor(np.array([[1, 2, 3]]))
c = torch.tensor(np.array([[4, 5, 6], [7, 8, 9]]))

d, e = list(zip((2, b), (3, c)))
print(d, e)
for i, boxes in enumerate(e):
    boxes[:, 0] = i
bboxes = torch.cat([a, b, c], 0)
print(bboxes)
