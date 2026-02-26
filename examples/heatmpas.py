import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image


def plot_heatmap(original_image, feature_map_tensor):
    # 1. 通道压缩 (这里演示简单平均法，Grad-CAM 会更复杂)
    # 去掉 batch 维度 -> [C, h, w], 在通道维度求平均 -> [h, w]
    heatmap_2d = torch.mean(feature_map_tensor.squeeze(0), dim=0)

    # 2. 上采样 (放大到原图尺寸)
    # 需要增加 batch 和 channel 维度才能进行 interpolate -> [1, 1, h, w]
    heatmap_2d = heatmap_2d.unsqueeze(0).unsqueeze(0)
    target_size = (original_image.shape[0], original_image.shape[1])  # 原图高宽
    heatmap_resized = F.interpolate(heatmap_2d, size=target_size, mode="bilinear", align_corners=False)

    # 移回 CPU 并转为 numpy -> [H, W]
    heatmap_np = heatmap_resized.squeeze().detach().cpu().numpy()

    # 3. 归一化到 [0, 1]
    heatmap_np = (heatmap_np - np.min(heatmap_np)) / (np.max(heatmap_np) - np.min(heatmap_np) + 1e-8)

    # 4. 转为伪彩色图 (使用 OpenCV 或 matplotlib)
    # 映射到 0-255 并转为 uint8
    heatmap_uint8 = np.uint8(255 * heatmap_np)
    # 应用 COLORMAP_JET (蓝到红)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    # OpenCV 默认是 BGR，转为 RGB以便 matplotlib 显示
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # 5. 叠加图片
    # 将原图转为灰度图作为背景，这步可选，但通常效果更好
    gray_img = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    gray_img_rgb = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)

    # 设定透明度 alpha
    alpha = 0.5
    superimposed_img = heatmap_color * alpha + gray_img_rgb * (1 - alpha)
    superimposed_img = np.uint8(superimposed_img)

    # 画图
    plt.imshow(superimposed_img)
    plt.axis("off")
    plt.show()


# 调用示例 (伪代码)
img = Image.open("/home/yangxf/Downloads/dv/3_co.jpg")
img = np.array(img)
fmap = torch.load("/home/yangxf/Downloads/feature_maps/c3.pt")
plot_heatmap(img, fmap)
