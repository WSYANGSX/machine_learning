import albumentations as A
from dataclasses import dataclass, MISSING


@dataclass
class AugCfg:
    """Basic augmentation configuration"""

    augs: list[A.BasicTransform] = MISSING
    bbox_params: A.BboxParams | None = None
    keypoint_params: A.KeypointParams | None = None
    additional_targets: str | None = None
    seed: int = 23
    probility: float = 1


# basic yolo enhancer
DEFAULT_YOLO_AUG = AugCfg(
    augs=[
        A.LongestMaxSize(max_size=416),
        A.HorizontalFlip(p=0.5),
        A.CoarseDropout(
            num_holes_range=(1, 10), hole_height_range=(10, 30), hole_width_range=(10, 30), fill="random_uniform", p=0.8
        ),
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1), p=0.8),
        A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.8),
        A.HueSaturationValue(hue_shift_limit=(-20, 20), sat_shift_limit=(-20, 20), val_shift_limit=(-20, 20), p=0.5),
        A.PadIfNeeded(min_height=416, min_width=416),
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["category_ids"],
        min_visibility=0.1,
        min_height=0.01,
        min_width=0.01,
        clip=True,
    ),
)


# basic multimodal enhancer
DEFAULT_YOLOMM_AUG = AugCfg(
    augs=[
        A.LongestMaxSize(max_size=416),
        A.HorizontalFlip(p=0.5),
        A.CoarseDropout(
            num_holes_range=(1, 10), hole_height_range=(10, 30), hole_width_range=(10, 30), fill="random_uniform", p=0.8
        ),
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1), p=0.8),
        A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.8),
        A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=0, val_shift_limit=(-20, 20), p=0.5),
        A.PadIfNeeded(min_height=416, min_width=416),
    ],
    bbox_params=A.BboxParams(
        format="pascal_voc",
        label_fields=["category_ids"],
        min_visibility=0.1,
        min_height=0.01,
        min_width=0.01,
        clip=True,
    ),
    additional_targets={"thermal": "image"},
)


# strong yolo enhancer
ENHANCED_YOLO_AUG = AugCfg(
    augs=[
        A.LongestMaxSize(max_size=416),
        A.HorizontalFlip(p=0.5),
        A.CoarseDropout(
            num_holes_range=(1, 10), hole_height_range=(10, 30), hole_width_range=(10, 30), fill="random_uniform", p=0.8
        ),
        A.Affine(translate_percent=(-0.1, 0.1), scale=(0.7, 1.8), p=0.8, keep_ratio=True),
        A.Sharpen(alpha=(0.3, 0.7), lightness=(0.3, 1.0), p=0.8),
        A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.8),
        A.HueSaturationValue(hue_shift_limit=(-30, 30), sat_shift_limit=(-30, 30), val_shift_limit=(-30, 30), p=0.8),
        A.PadIfNeeded(min_height=416, min_width=416, position="center"),
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["category_ids"],
        min_visibility=0.1,
        min_height=0.01,
        min_width=0.01,
        clip=True,
    ),
)
