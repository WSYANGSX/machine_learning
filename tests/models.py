import torch

# x, y = torch.meshgrid(torch.arange(4), torch.arange(6), indexing="xy")

# print(x, y)
# # print(torch.stack([x, y], dim=-1))


# grid_x, grid_y = torch.meshgrid(torch.tensor([1, 2, 3]), torch.tensor([4, 5]), indexing="xy")
# grid_xy = torch.stack((grid_x, grid_y), dim=-1).float()  # [H, W, 2]
# print(grid_x, grid_y)
# print(grid_xy)

# grid_y, grid_x = torch.meshgrid(
#     torch.tensor([1, 2, 3]),
#     torch.tensor([4, 5]),
#     indexing="ij",  # 注意参数顺序
# )

# # 然后堆叠为(x,y)
# grid_xy = torch.stack((grid_x, grid_y), dim=-1)  # [H, W, 2]
# print(grid_xy)


a = torch.rand((2, 3, 4))
b = a[..., 1:2]
print(b)
