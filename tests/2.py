import torch

pos_x = torch.randn(1, 5, 5, 64)  # [B, H, W, NumPosFeats]
pos_x_sin = pos_x[:, :, :, 0::2].sin()  # [B, H, W, NumPosFeats//2]
pos_x_cos = pos_x[:, :, :, 1::2].cos()  # [B, H, W, NumPosFeats//2]
print(pos_x_sin.shape)  # [B, H, W, NumPosFeats//2]
print(pos_x_cos.shape)  # [B, H, W, NumPosFeats//2]
pos_x = torch.stack((pos_x_sin, pos_x_cos), dim=4).flatten(3)  # [B, H, W, NumPosFeats//2, 2] -> [B, H, W, NumPosFeats]
print(pos_x.shape)  # [B, H, W, NumPosFeats//2, 2]
