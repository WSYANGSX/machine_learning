from typing import Sequence, Dict, Any, Optional, Union

import torch
import numpy as np
import albumentations as A

from machine_learning.utils.cfg import BaseCfg
from dataclasses import dataclass, MISSING


@dataclass
class AugCfg(BaseCfg):
    """Basic augmentation configuration"""

    augs: list[A.BasicTransform] = MISSING
    bbox_params: A.BboxParams | None = None
    keypoint_params: A.KeypointParams | None = None
    additional_targets: str | None = None
    seed: int = 23
    probility: float = 1


# default augcfg, none enhancer
DEFAULT_AUG = AugCfg(
    augs=[
        A.HorizontalFlip(p=0.5),
        A.CoarseDropout(
            num_holes_range=(1, 10),
            hole_height_range=(10, 30),
            hole_width_range=(10, 30),
            fill="random_uniform",
            p=0.8,
        ),
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1), p=0.8),
        A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.8),
    ],
)

# basic yolo enhancer
DEFAULT_YOLO_AUG = AugCfg(
    augs=[
        A.LongestMaxSize(max_size=416),
        A.HorizontalFlip(p=0.5),
        A.CoarseDropout(
            num_holes_range=(1, 10),
            hole_height_range=(10, 30),
            hole_width_range=(10, 30),
            fill="random_uniform",
            p=0.8,
        ),
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1), p=0.8),
        A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.8),
        A.HueSaturationValue(
            hue_shift_limit=(-20, 20),
            sat_shift_limit=(-20, 20),
            val_shift_limit=(-20, 20),
            p=0.5,
        ),
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
            num_holes_range=(1, 10),
            hole_height_range=(10, 30),
            hole_width_range=(10, 30),
            fill="random_uniform",
            p=0.8,
        ),
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1), p=0.8),
        A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.8),
        A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=0, val_shift_limit=(-20, 20), p=0.5),
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
    additional_targets={"thermal": "image"},
)


# strong yolo enhancer
ENHANCED_YOLO_AUG = AugCfg(
    augs=[
        A.LongestMaxSize(max_size=416),
        A.HorizontalFlip(p=0.5),
        A.CoarseDropout(
            num_holes_range=(1, 10),
            hole_height_range=(10, 30),
            hole_width_range=(10, 30),
            fill="random_uniform",
            p=0.8,
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


class TransformBase:
    def __init__(self, cfg: AugCfg):
        pass

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        pass


class ImgTransform(TransformBase):
    "Unified data transformation base class, supporting classification and detection tasks"

    def __init__(
        self,
        aug_cfg: AugCfg,
        normalize: bool = True,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
        to_tensor: bool = True,
    ):
        """
        Initialization converter

        Args:
            augmentation: Albumentations enhance combination.
            to_tensor: Whether to convert to Tensor.
            normalize: Whether to standardize the image mean.
            mean: standardized mean.
            std: standardized standard deviation.
            bbox_params: Bounding box parameters (for detection tasks only).
            keypoint_params: Key point parameters, used for pose estimation.
        """
        self.aug_cfg = aug_cfg

        self.normalize = normalize
        if self.normalize:
            self.mean = mean or (0.0, 0.0, 0.0)
            self.std = std or (1.0, 1.0, 1.0)

        self.to_tensor = to_tensor

        # Build a conversion pipeline
        self.augmentation_pipeline = self._build_augmentation_pipeline(self.aug_cfg.augs)
        self.postprocess_pipeline = self._build_postprocess_pipeline()

    def _build_augmentation_pipeline(self, augmentation: list[A.BasicTransform]) -> Optional[A.Compose]:
        """Building the data augmentation pipeline"""

        return A.Compose(
            transforms=augmentation,
            bbox_params=self.aug_cfg.bbox_params,
            keypoint_params=self.aug_cfg.keypoint_params,
            additional_targets=self.aug_cfg.additional_targets,
            p=self.aug_cfg.probility,
            seed=self.aug_cfg.seed,
        )

    def _build_postprocess_pipeline(self) -> Dict[str, Any]:
        postprocess = []

        if self.normalize:
            postprocess.append(A.Normalize(self.mean, self.std))

        if self.to_tensor:
            postprocess.append(A.pytorch.ToTensorV2())

        return A.Compose(
            transforms=postprocess,
            bbox_params=self.aug_cfg.bbox_params,
            keypoint_params=self.aug_cfg.keypoint_params,
            additional_targets=self.aug_cfg.additional_targets,
            p=self.aug_cfg.probility,
            seed=self.aug_cfg.seed,
        )

    def _apply_augmentation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Application Data Augmentation"""
        _necessary_data_type = ["image"]

        for data_type in _necessary_data_type:
            if data_type not in data:
                raise ValueError("Necessary data type {data_type} not in input data.")

        transform_args = {"image": data["image"]}

        # add parameters
        if "mask" in data:
            transform_args["mask"] = data["mask"]

        if "bboxes" in data and self.aug_cfg.bbox_params:
            transform_args["bboxes"] = data["bboxes"]
            if "category_ids" in data:
                transform_args["category_ids"] = data["category_ids"]

        if "keypoints" in data and self.aug_cfg.keypoint_params:
            transform_args["keypoints"] = data["keypoints"]

        if self.aug_cfg.additional_targets:
            for key in self.aug_cfg.additional_targets:
                if key in data:
                    transform_args[key] = data[key]

        # apply transform
        transformed_data = self.augmentation_pipeline(**transform_args)

        return transformed_data

    def __call__(self, data: Dict[str, Any], augment: bool = True) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        if augment:
            data = self._apply_augmentation(data)

        data = self.postprocess_pipeline(**data)

        return data
