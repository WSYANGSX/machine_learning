import torch

a = torch.randn((3, 3, 2))
print(a[:2, 2, :2])
# b = torch.randn((3, 3, 2))
# print(b)
# c = torch.max(a, b)
# print(c)
# d = c.max(2)
# j = d[0] < 0.5
# print(j)
# print(a[j])
