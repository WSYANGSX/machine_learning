import torch

# x = torch.tensor(
#     [
#         [[0.1, 0.5, 0.6], [0.2, 0.3, 0.45], [0.25, 0.5, 0.19]],
#         [[0.02, 0.04, 0.33], [0.21, 0.36, 0.25], [0.15, 0.11, 0.30]],
#         [[0.08, 0.5, 0.23], [0.14, 0.3, 0.55], [0.243, 0.465, 0.19]],
#     ]
# )
# indices = (
#     torch.tensor([0, 2], dtype=torch.int64),
#     torch.tensor([0, 1], dtype=torch.int64),
#     torch.tensor([0, 2], dtype=torch.int64),
# )
# print(
#     x[
#         torch.tensor([0, 2], dtype=torch.int64),
#         torch.tensor([0, 1], dtype=torch.int64),
#     ]
# )
# print(x[0:2, 0:2, 0:2])

x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
# x[:, [0, 1, 1, 2]] = 0
# print(x)
x[range(4), [0, 1, 2, 2]] = 0
print(x)
