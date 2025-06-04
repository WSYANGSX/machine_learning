import torch
import numpy as np


def rescale_boxes(boxes: torch.Tensor, current_dim: int, original_shape: tuple[int]) -> torch.Tensor:
    """
    将目标检测模型输出的边界框坐标从调整后的正方形图像尺寸转换回原始图像尺寸,
    [example](/home/yangxf/WorkSpace/machine_learning/docs/pictures/01.jpg)
    """
    orig_h, orig_w = original_shape

    # 计算增加的pad, 应对pad后放缩的情况
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))

    # 移除pad后的尺寸
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x

    # 重新映射边界框
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h

    return boxes


def xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2

    return y


def xywh2xyxy_np(x):
    y = np.zeros_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def xyxy2xywh(x: torch.Tensor) -> torch.Tensor:
    y = x.new(x.shape)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]

    return y


def xyxy2xywh_np(x: torch.Tensor) -> torch.Tensor:
    y = np.zeros_like(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]

    return y


def to_absolute_labels(img: torch.Tensor | np.ndarray, bboxes: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    img, bboxes = img, bboxes
    h, w, _ = img.shape
    bboxes[:, [0, 2]] *= w
    bboxes[:, [1, 3]] *= h
    return bboxes


def to_relative_labels(img: torch.Tensor | np.ndarray, bboxes: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    img, bboxes = img, bboxes
    h, w, _ = img.shape
    bboxes[:, [0, 2]] /= w
    bboxes[:, [1, 3]] /= h
    return bboxes


def yolo2voc(img: torch.Tensor | np.ndarray, bboxes: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    return to_absolute_labels(img, xywh2xyxy_np(bboxes))
