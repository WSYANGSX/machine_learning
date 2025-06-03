import numpy as np

import albumentations as A
from albumentations import DualTransform


class PadShortEdge(DualTransform):
    def __init__(self, p=0.5):
        super().__init__(p)

    def get_params_dependent_on_data(self, params, data):
        height, width = data["image"].shape[:2]
        return {"height": height, "width": width}

    def apply(self, img, *args, **params):
        h, w = params["channels"], params["height"], params["width"]
        dim_diff = np.abs(h - w)
        # 填充数值
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # 填充数 (左， 右， 上， 下， 前， 后)
        pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)

        img = np.pad(img, pad, "constant", value=pad_value)

        return img

    def apply_to_bboxes(self, bboxes, *args, **params):
        pass


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
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4),
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
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.3),  # 允许更小的bbox可见比例
)
