from typing import Literal

import torch
import numpy as np


def rescale_padded_boxes(boxes: torch.Tensor, current_dim: int, original_shape: tuple[int]) -> torch.Tensor:
    """
    将目标检测模型输出的边界框坐标从padding后的正方形图像尺寸转换回原始图像尺寸,
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


def bbox_iou(
    bbox1: torch.Tensor,
    bbox2: torch.Tensor,
    bbox_format: Literal["pascal_voc", "coco"] = "pascal_voc",
    iou_type: Literal["default", "giou", "diou", "ciou"] = "Default",
    eps: float = 1e-9,
):
    """Returns the IoU of box1 to box2. box1 is 4, box2 is nx4"""
    # Get the coordinates of bounding boxes
    if bbox_format == "coco":
        bbox1, bbox2 = xywh2xyxy(bbox1), xywh2xyxy(bbox2)

    bbox2 = bbox2.T

    bb1_x1, bb1_y1, bb1_x2, bb1_y2 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
    bb2_x1, bb2_y1, bb2_x2, bb2_y2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]

    # Intersection area
    inter = (torch.min(bb1_x2, bb2_x2) - torch.max(bb1_x1, bb2_x1)).clamp(0) * (
        torch.min(bb1_y2, bb2_y2) - torch.max(bb1_y1, bb2_y1)
    ).clamp(0)

    # Union Area
    w1, h1 = bb1_x2 - bb1_x1, bb1_y2 - bb1_y1 + eps
    w2, h2 = bb2_x2 - bb2_x1, bb2_y2 - bb2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if iou_type == "giou" or iou_type == "diou" or iou_type == "ciou":
        # convex (smallest enclosing box) width
        cw = torch.max(bb1_x2, bb2_x2) - torch.min(bb1_x1, bb2_x1)
        ch = torch.max(bb1_y2, bb2_y2) - torch.min(bb1_y1, bb2_y1)
        if iou_type == "diou" or iou_type == "ciou":
            c2 = cw**2 + ch**2 + eps
            rho2 = ((bb2_x1 + bb2_x2 - bb1_x1 - bb1_x2) ** 2 + (bb2_y1 + bb2_y2 - bb1_y1 - bb1_y2) ** 2) / 4
            if iou_type == "diou":
                return iou - rho2 / c2  # DIoU
            elif iou_type == "ciou":
                v = (4 / torch.pi**2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                return iou - (rho2 / c2 + v * alpha)
        else:
            c_area = cw * ch + eps
            return iou - (c_area - union) / c_area
    else:
        return iou
