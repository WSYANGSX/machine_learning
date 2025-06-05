import torch
import numpy as np

a = torch.tensor(np.array([]))
b = torch.tensor(np.array([[1, 2, 3]]))
c = torch.tensor(np.array([[4, 5, 6], [7, 8, 9]]))

d, e = list(zip((1, a), (2, b), (3, c)))
print(d, e)
for i, boxes in enumerate(e):
    boxes[:, 0] = i
bboxes = torch.cat(e, 0)
print(bboxes)
