from typing import Sequence, Dict, Any, Optional, Union

import torch
import numpy as np
import albumentations as A
from torchvision import transforms as T


class CustomTransform:
    "Unified data transformation base class, supporting classification and detection tasks"

    def __init__(
        self,
        augmentation: Sequence[A.BasicTransform] | None = None,
        to_tensor: bool = True,
        normalize: bool = True,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
        bbox_params: Optional[A.BboxParams] = None,
        keypoint_params: Optional[A.KeypointParams] = None,
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
        self.to_tensor = to_tensor
        self.normalize = normalize
        self.mean = mean or (0.0, 0.0, 0.0)
        self.std = std or (1.0, 1.0, 1.0)

        self.bbox_params = bbox_params
        self.keypoint_params = keypoint_params

        # Build a conversion pipeline
        self.augmentation_pipeline = self._build_augmentation_pipeline(augmentation)

    def _build_augmentation_pipeline(self, augmentation: Optional[Sequence[A.BasicTransform]]) -> Optional[A.Compose]:
        """Building the data augmentation pipeline"""
        if not augmentation:
            return None

        return A.Compose(augmentation, bbox_params=self.bbox_params, keypoint_params=self.keypoint_params)

    def _apply_augmentation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Application Data Augmentation"""
        transform_args = {"image": data["image"]}

        # add parameters
        if "mask" in data:
            transform_args["mask"] = data["mask"]

        if "bboxes" in data and self.bbox_params:
            transform_args["bboxes"] = data["bboxes"]
            if "category_ids" in data:
                transform_args["category_ids"] = data["category_ids"]

        if "keypoints" in data and self.keypoint_params:
            transform_args["keypoints"] = data["keypoints"]

        # apply transform
        transformed = self.augmentation_pipeline(**transform_args)

        data["image"] = transformed["image"]
        if "bboxes" in transformed:
            data["bboxes"] = transformed["bboxes"]
        if "category_ids" in transformed:
            data["category_ids"] = transformed["category_ids"]
        if "mask" in transformed:
            data["mask"] = transformed["mask"]
        if "keypoints" in transformed:
            data["keypoints"] = transformed["keypoints"]

        return data

    def _apply_postprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.to_tensor:
            transform_args = {"image": data["image"]}
            if "mask" in data:
                transform_args["mask"] = data["mask"]

            transformed = A.pytorch.ToTensorV2(transpose_mask=True)(**transform_args)

            data["image"] = transformed["image"]
            if "mask" in transformed:
                data["mask"] = transformed["mask"]

            fields_to_convert = ["labels", "bboxes", "category_ids", "keypoints"]
            for field in fields_to_convert:
                if field in data and not isinstance(data[field], torch.Tensor):
                    dtype = torch.float32 if field in ["bboxes", "keypoints"] else torch.int64
                    data[field] = torch.tensor(data[field], dtype=dtype)

        if self.normalize:
            if isinstance(data["image"], torch.Tensor):
                if data["image"].dtype != torch.float32:
                    data["image"] = data["image"].float() / 255.0

                normalize = T.Normalize(mean=self.mean, std=self.std)
                data["image"] = normalize(data["image"])

            elif isinstance(data["image"], np.ndarray):
                if data["image"].dtype != np.float32:
                    data["image"] = data["image"].astype(np.float32) / 255.0

                mean = np.array(self.mean, dtype=np.float32)
                std = np.array(self.std, dtype=np.float32)
                data["image"] = (data["image"] - mean) / std

        return data

    def __call__(self, data: Dict[str, Any], augment: bool = True) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        if self.augmentation_pipeline and augment:
            data = self._apply_augmentation(data)

        data = self._apply_postprocess(data)

        return data
