import torch

grid_y, grid_x = torch.meshgrid(torch.arange(15), torch.arange(12), indexing="ij")
print(grid_x)
print(grid_y)

grid_xy = torch.stack((grid_x, grid_y), dim=-1)  # [H, W, 2]
print(grid_xy)
