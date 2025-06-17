import numpy as np
from typing import Sequence

import albumentations as A
from albumentations import DualTransform


class PadShortEdge(DualTransform):
    def __init__(self, pad_values: int | Sequence[tuple[int]], p: float = 0.5):
        """Pad the image to square along short side.

        Args:
            pad_values (int | Sequence[tuple[int]]): sequence or scalar. The values to set the padded values for each
            axis. ((before_1, after_1), ... (before_N, after_N)) unique pad constants for each axis. (before, after) or
            ((before, after),) yields same before and after constants for each axis. (constant,) or constant is a
            shortcut for before = after = constant for all axes. Default is 0.
            p (float, optional): probability to use this transform. Defaults to 0.5.
        """
        super().__init__(p)
        self.pad_values = pad_values

    def get_params_dependent_on_data(self, params, data):
        img = data["image"]
        # 统一获取高度和宽度，不依赖通道数
        height, width = img.shape[:2]
        return {"height": height, "width": width}

    def apply(self, img, *args, **params):
        h, w = params["height"], params["width"]
        dim_diff = np.abs(h - w)
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2

        # 处理不同维度的图像
        if img.ndim == 2:  # 灰度图 (H, W)
            pad_width = [(pad1, pad2), (0, 0)] if h <= w else [(0, 0), (pad1, pad2)]
        elif img.ndim == 3:  # 彩色图 (H, W, C)
            pad_width = [(pad1, pad2), (0, 0), (0, 0)] if h <= w else [(0, 0), (pad1, pad2), (0, 0)]
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")

        # 确保pad_values兼容
        if isinstance(self.pad_values, int):
            constant_values = self.pad_values
        else:
            constant_values = self.pad_values

        return np.pad(img, pad_width, "constant", constant_values=constant_values)

    def apply_to_bboxes(self, bboxes, *args, **params):
        h, w = params["height"], params["width"]

        dim_diff = abs(h - w)
        pad1 = dim_diff // 2

        # 防止除零错误
        epsilon = 1e-8

        if h <= w:
            total_padded_height = w

            # 变换y坐标：y_min和y_max
            bboxes[:, 1] = (bboxes[:, 1] * h + pad1) / (total_padded_height + epsilon)
            bboxes[:, 3] = (bboxes[:, 3] * h + pad1) / (total_padded_height + epsilon)

        else:
            total_padded_width = h

            bboxes[:, 0] = (bboxes[:, 0] * w + pad1) / (total_padded_width + epsilon)
            bboxes[:, 2] = (bboxes[:, 2] * w + pad1) / (total_padded_width + epsilon)

        bboxes = np.clip(bboxes, 0, 1)
        return bboxes


# 基础增强
DEFAULT_AUG = A.Compose(
    [
        # 锐化（强度适中）
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1), p=0.8),
        # 仿射变换（平移+缩放，保持宽高比）
        A.Affine(translate_percent=(-0.08, 0.08), scale=(0.8, 1.5), p=0.8, keep_ratio=True),
        # 亮度对比度调整
        A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.8),
        # 色调饱和度调整
        A.HueSaturationValue(hue_shift_limit=(-20, 20), sat_shift_limit=(-20, 20), val_shift_limit=(-20, 20), p=0.5),
        # 水平翻转
        A.HorizontalFlip(p=0.5),
        PadShortEdge(pad_values=0.1, p=1),
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        min_width=0.02,
        min_height=0.02,
        min_visibility=0.3,
        label_fields=["category_ids"],
        clip=True,
    ),
)

# 强增强
ENHANCED_AUG = A.Compose(
    [
        A.CoarseDropout(
            num_holes_range=(1, 10), hole_height_range=(10, 30), hole_width_range=(10, 30), fill="random_uniform", p=0.8
        ),
        # 更强烈的锐化
        A.Sharpen(alpha=(0.3, 0.7), lightness=(0.3, 1.0), p=0.8),
        # 更大幅度的仿射变换
        A.Affine(translate_percent=(-0.1, 0.1), scale=(0.7, 1.8), p=0.8, keep_ratio=True),
        # 更强烈的亮度对比度调整
        A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.8),
        # 更大幅度的色调偏移
        A.HueSaturationValue(hue_shift_limit=(-30, 30), sat_shift_limit=(-30, 30), val_shift_limit=(-30, 30), p=0.8),
        # 强制水平翻转（100%概率）
        A.HorizontalFlip(p=1.0),
        PadShortEdge(pad_values=0.1, p=1),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.3, label_fields=["category_ids"], clip=True),
)
