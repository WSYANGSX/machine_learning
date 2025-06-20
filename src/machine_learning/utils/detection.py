from typing import Literal

import torch
import numpy as np


def rescale_padded_boxes(boxes: torch.Tensor, current_dim: int, original_shape: tuple[int]) -> torch.Tensor:
    """
    将目标检测模型输出的边界框坐标从padding后的正方形图像尺寸转换回原始图像尺寸,
    [example](/home/yangxf/WorkSpace/machine_learning/docs/pictures/01.jpg)
    """
    _, orig_h, orig_w = original_shape

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
    iou_type: Literal["default", "giou", "diou", "ciou"] = "default",
    eps: float = 1e-9,
    safe: bool = True,
) -> torch.Tensor:
    """计算改进的 IoU, 增强数值稳定性"""

    # 0. 输入验证
    if safe:
        # 检查输入是否为有限值
        if torch.isnan(bbox1).any():
            raise ValueError("输入边界框 bbox1 包含非有限值(NaN)")
        elif torch.isinf(bbox1).any():
            raise ValueError("输入边界框 bbox1 包含非有限值(inf)")

        if torch.isnan(bbox2).any():
            raise ValueError("输入边界框 bbox2 包含非有限值(NaN)")
        elif torch.isinf(bbox2).any():
            raise ValueError("输入边界框 bbox2 包含非有限值(inf)")

        # 检查坐标有效性
        if bbox_format == "pascal_voc":
            if (bbox1[2] < bbox1[0]).any() or (bbox1[3] < bbox1[1]).any():
                raise ValueError("bbox1 包含无效坐标 (x2 < x1 或 y2 < y1)")
            if (bbox2[2] < bbox2[0]).any() or (bbox2[3] < bbox2[1]).any():
                raise ValueError("bbox2 包含无效坐标 (x2 < x1 或 y2 < y1)")

    # 1. 格式转换
    if bbox_format == "coco":
        bbox1 = xywh2xyxy(bbox1)
        bbox2 = xywh2xyxy(bbox2)

    # 2. 提取坐标 - 添加维度处理
    if bbox1.dim() == 1:
        bbox1 = bbox1.unsqueeze(0)
    if bbox2.dim() == 1:
        bbox2 = bbox2.unsqueeze(0)

    bb1_x1, bb1_y1, bb1_x2, bb1_y2 = bbox1[:, 0], bbox1[:, 1], bbox1[:, 2], bbox1[:, 3]
    bb2_x1, bb2_y1, bb2_x2, bb2_y2 = bbox2[:, 0], bbox2[:, 1], bbox2[:, 2], bbox2[:, 3]

    # 3. 计算交集区域 (添加保护性clamp)
    inter_x1 = torch.max(bb1_x1, bb2_x1)
    inter_y1 = torch.max(bb1_y1, bb2_y1)
    inter_x2 = torch.min(bb1_x2, bb2_x2)
    inter_y2 = torch.min(bb1_y2, bb2_y2)

    inter_width = (inter_x2 - inter_x1).clamp(min=0)
    inter_height = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_width * inter_height

    # 4. 计算并集区域 (所有维度添加eps)
    bb1_width = (bb1_x2 - bb1_x1).clamp(min=eps)
    bb1_height = (bb1_y2 - bb1_y1).clamp(min=eps)
    bb2_width = (bb2_x2 - bb2_x1).clamp(min=eps)
    bb2_height = (bb2_y2 - bb2_y1).clamp(min=eps)

    union_area = (bb1_width * bb1_height) + (bb2_width * bb2_height) - inter_area + eps

    # 5. 基本IoU计算
    iou = inter_area / union_area

    # 6. 高级IoU变体
    if iou_type != "default":
        # 最小包围框
        c_x1 = torch.min(bb1_x1, bb2_x1)
        c_y1 = torch.min(bb1_y1, bb2_y1)
        c_x2 = torch.max(bb1_x2, bb2_x2)
        c_y2 = torch.max(bb1_y2, bb2_y2)

        c_width = (c_x2 - c_x1).clamp(min=eps)
        c_height = (c_y2 - c_y1).clamp(min=eps)
        c_area = c_width * c_height

        if iou_type in ["diou", "ciou"]:
            # 中心点距离平方
            bb1_cx = (bb1_x1 + bb1_x2) / 2
            bb1_cy = (bb1_y1 + bb1_y2) / 2
            bb2_cx = (bb2_x1 + bb2_x2) / 2
            bb2_cy = (bb2_y1 + bb2_y2) / 2

            center_dist_sq = (bb2_cx - bb1_cx).pow(2) + (bb2_cy - bb1_cy).pow(2)
            c_diagonal_sq = c_width.pow(2) + c_height.pow(2) + eps

            diou_term = center_dist_sq / c_diagonal_sq

            if iou_type == "ciou":
                # 改进的宽高比计算 (避免除零)
                arctan_diff = torch.atan(bb2_width / bb2_height.clamp(min=eps)) - torch.atan(
                    bb1_width / bb1_height.clamp(min=eps)
                )

                # 更稳定的v计算
                v = (4 / (torch.pi**2)) * arctan_diff.pow(2)

                # 动态加权
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)

                return iou - diou_term - v * alpha
            else:
                return iou - diou_term
        else:  # GIoU
            return iou - (c_area - union_area) / c_area

    return iou
