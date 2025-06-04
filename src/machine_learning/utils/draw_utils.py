import cv2
import math
import torch
import numpy as np

from typing import Sequence, Mapping
import matplotlib.pyplot as plt


BOX_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (255, 255, 255)  # White


def add_bbox(
    img: np.ndarray | torch.Tensor,
    bbox: np.ndarray | torch.Tensor,
    class_name: str,
    color: tuple[int] = BOX_COLOR,
    thickness: int = 2,
) -> np.ndarray | torch.Tensor:
    """向图像中添加单个边界框（修复顶部标签显示问题）

    Args:
        img (np.ndarray | torch.Tensor): 要添加边界框的图像
        bbox (np.ndarray | torch.Tensor): 边界框参数, Pascal VOC格式(x_min, y_min, x_max, y_max)
        class_name (str): 边界框中物体的类别名称
        color (tuple[int], optional): 边界框颜色，默认红色
        thickness (int, optional): 边界框的粗细, 默认2

    Returns:
        np.ndarray | torch.Tensor: 添加边界框后的图像
    """
    # 确保处理的是numpy数组（如果是Tensor则转换）
    if isinstance(img, torch.Tensor):
        img_np = img.cpu().numpy().transpose(1, 2, 0) if img.dim() == 3 else img.cpu().numpy()
        return_tensor = True
    else:
        img_np = img.copy()
        return_tensor = False

    # 获取图像尺寸
    img_height, img_width = img_np.shape[:2]

    # 转换坐标为整数
    x_min, y_min, x_max, y_max = map(int, bbox)

    # 绘制边界框
    cv2.rectangle(img_np, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    # 获取文本尺寸
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    label_height = int(1.3 * text_height)

    # 智能调整标签位置（防止超出图像边界）
    if y_min - label_height < 0:  # 顶部空间不足
        # 将标签放在边界框内部底部
        label_y_top = y_min
        label_y_bottom = label_y_top + label_height
        text_y = label_y_top + label_height - int(0.3 * text_height)
    else:  # 正常情况：标签放在边界框上方
        label_y_top = y_min - label_height
        label_y_bottom = y_min
        text_y = y_min - int(0.3 * text_height)

    # 确保标签不超出图像底部
    if label_y_bottom > img_height:
        label_y_top = max(0, img_height - label_height)
        label_y_bottom = img_height
        text_y = label_y_bottom - int(0.3 * text_height)

    # 绘制标签背景和文本
    cv2.rectangle(img_np, (x_min, label_y_top), (x_min + text_width, label_y_bottom), BOX_COLOR, -1)
    cv2.putText(
        img_np,
        text=class_name,
        org=(x_min, text_y),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )

    # 如果需要，转换回Tensor
    if return_tensor:
        img = torch.from_numpy(img_np.transpose(2, 0, 1)) if img_np.ndim == 3 else torch.from_numpy(img_np)
    else:
        img = img_np

    return img


def visualize_bboxes(
    image: np.ndarray | torch.Tensor,
    bboxes: np.ndarray | torch.Tensor,
    category_ids: Sequence[int],
    category_id_to_name: Sequence[str] | Mapping[int, str],
) -> None:
    """显示添加边界框后的图像

    Args:
        image (np.ndarray | torch.Tensor): 要添加边界框的图像.
        bboxes (np.ndarray | torch.Tensor): 边界框, 以pascal_voc格式输入(x_min, y_min, x_max, y_max).
        category_ids (Sequence[int]): 边界框中物体的类别编号序列.
        category_id_to_name (Sequence[str]): 边界框中物体的类别编号对应的名称序列.
    """
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = add_bbox(img, bbox, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis("off")
    plt.imshow(img)
    plt.show()


def draw_figures(figures: torch.Tensor | np.ndarray, cmap: str):
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
