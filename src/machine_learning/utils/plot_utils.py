import math
import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_raw_recon_figures(raw_figures: torch.Tensor | np.ndarray, recon_figures: torch.Tensor | np.ndarray):
    plt.figure(figsize=(10, 4))
    num_figures = len(raw_figures)

    for i in range(len(raw_figures)):
        # 原始图像
        ax = plt.subplot(2, num_figures, i + 1)
        plt.imshow(raw_figures[i].cpu().squeeze(), cmap="gray")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # 重构图像
        ax = plt.subplot(2, num_figures, i + 1 + num_figures)
        plt.imshow(recon_figures[i].cpu().squeeze(), cmap="gray")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()


def plot_figures(figures: torch.Tensor | np.ndarray, cmap: str):
    if isinstance(figures, torch.Tensor):
        figures = figures.cpu().numpy()

    if figures.ndim == 3:
        figures = figures[None, ...]

    plt.figure(figsize=(10, 10))

    figures_num = len(figures)
    col_num = 4
    row_num = math.ceil(figures_num / 4)

    for row in range(row_num):
        for col in range(col_num):
            # 计算当前图像的索引
            index = row * col_num + col
            if index < figures_num:
                ax = plt.subplot(row_num, col_num, index + 1)
                plt.imshow(figures[index].squeeze(), cmap=cmap)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.show()
