from typing import Literal, Sequence

import torch
import numpy as np
import albumentations as A
from torchvision import transforms

from machine_learning.utils.augmentations import DEFAULT_AUG, ENHANCED_AUG


class BaseTransform:
    r"""Base transform class to encapsulate the interfaces of albumentations.Compose and transforms.Compose."""

    def __init__(
        self,
        augmentation: Literal["default", "enhanced"] | A.Compose | None = None,
        to_tensor: bool = True,
        normalize: bool | None = True,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
    ):
        if augmentation is not None:
            if isinstance(augmentation, str):
                self.augmentation = DEFAULT_AUG if augmentation == "default" else ENHANCED_AUG
            else:
                self.augmentation = augmentation
        else:
            self.augmentation = None

        if normalize:
            self.normalize = transforms.Normalize(mean=mean, std=std)

        if to_tensor:
            self.to_tensor = transforms.ToTensor()  # It can only be applied in 2/3 dimensions

    def __call__(self, data: Sequence[np.ndarray], augment: bool = True) -> tuple[torch.Tensor]:
        """The specific logical implementation of data enhancement"""
        pass


class YoloTransform(BaseTransform):
    def __init__(
        self,
        augmentation: Literal["default", "enhanced"] | A.Compose | None = None,
        to_tensor: bool = True,
        normalize: bool | None = True,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
    ):
        """
        Implementation of data transform for Yolo object detection algorithm

        Args:
            augmentation (Literal[&quot;default&quot;, &quot;enhanced&quot;], A.Compose, optional): Enhancer.
            normalize (bool, optional): Whether to normalize the data. Defaults to True.
            mean (Sequence[float] | None, optional): If the normalize parameter set to True, the mean value needs to be
            provided. Defaults to None.
            std (Sequence[float] | None, optional): If the normalize parameter set to True, the std value needs to be
            provided. Defaults to None.
            to_tensor (bool, optional): Whether to convert data to Tensor. Defaults to True.
        """
        super().__init__(augmentation, to_tensor, normalize, mean, std)

    def __call__(self, data: Sequence[np.ndarray], augment: bool = True) -> tuple[torch.Tensor]:
        """
        Apply data augmentation.

        Args:
            data: Tuples containing (image, bboxes, category_ids)

        Returns:
            Converted tensors (image_tensor, bboxes_tensor, category_ids)
        """
        img, bboxes, category_ids = data

        if self.augmentation is not None and augment:
            auged_data = self.augmentation(image=img, bboxes=bboxes, category_ids=category_ids)

            img = auged_data["image"]
            bboxes = auged_data["bboxes"]
            category_ids = auged_data[
                "category_ids"
            ]  # augementation will convert other unregistered data types to list types

        # 转化为Tensor
        if self.to_tensor:
            img = self.to_tensor(img)
            bboxes = torch.from_numpy(bboxes)
            category_ids = torch.tensor(category_ids)

        # 归一化
        if self.normalize:
            img = self.normalize(img)

        return img, bboxes, category_ids
