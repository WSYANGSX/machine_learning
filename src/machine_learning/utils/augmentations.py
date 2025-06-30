import albumentations as A


# basic enhancer
DEFAULT_YOLO_AUG = A.Compose(
    [
        A.LongestMaxSize(max_size=416, area_for_downscale="image"),
        A.HorizontalFlip(p=0.5),
        A.Affine(translate_percent=(-0.08, 0.08), scale=(0.8, 1.5), p=0.8, keep_ratio=True),
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1), p=0.8),
        A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.8),
        A.HueSaturationValue(hue_shift_limit=(-20, 20), sat_shift_limit=(-20, 20), val_shift_limit=(-20, 20), p=0.5),
        A.PadIfNeeded(min_height=416, min_width=416, position="center"),
    ],
    bbox_params=A.BboxParams(format="yolo", label_fields=["category_ids"], clip=True),
    strict=True,
)

# strong enhancer
ENHANCED_YOLO_AUG = A.Compose(
    [
        A.LongestMaxSize(max_size=416, area_for_downscale="image"),
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
    bbox_params=A.BboxParams(format="yolo", label_fields=["category_ids"], clip=True),
    strict=True,
)
