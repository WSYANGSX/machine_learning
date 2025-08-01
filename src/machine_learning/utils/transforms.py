from typing import Sequence, Dict, Any, Optional, Union

import torch
import numpy as np
import albumentations as A

from machine_learning.utils.aug import AugCfg


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
