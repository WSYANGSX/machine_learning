from typing import Sequence

import torch
import numpy as np
import albumentations as A
from torchvision import transforms


class TransformBase:
    r"""Base transform class to encapsulate the interfaces of albumentations.Compose and transforms.Compose."""

    def __init__(
        self,
        augmentation: A.Compose | None = None,
        to_tensor: bool = True,
        normalize: bool | None = True,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
    ):
        self.augmentation = augmentation

        if to_tensor:
            self.to_tensor = transforms.ToTensor()  # It can only be applied in 2/3 dimensions

        if normalize:
            self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, data: dict[str, np.ndarray], augment: bool = True) -> tuple[torch.Tensor]:
        """The specific logical implementation of data enhancement"""
        data = data["image"]
        labels = data["labels"]

        if self.augmentation is not None and augment:
            data = self.augmentation(data)

        # convert to tensor
        if self.to_tensor:
            data = self.to_tensor(data)
            labels = torch.tensor(labels)

        # normalize
        if self.normalize:
            data = self.normalize(data)

        return {"data": data, "labels": labels}


class YoloTransform(TransformBase):
    def __init__(
        self,
        augmentation: A.Compose | None = None,
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

    def __call__(self, data: dict[str, np.ndarray], augment: bool = True) -> tuple[torch.Tensor]:
        img = data["image"]
        bboxes = data["bboxes"]
        category_ids = data["category_ids"]

        if self.augmentation is not None and augment:
            auged_data = self.augmentation(image=img, bboxes=bboxes, category_ids=category_ids)

            img = auged_data["image"]
            bboxes = auged_data["bboxes"]
            category_ids = auged_data[
                "category_ids"
            ]  # augementation will convert other unregistered data types to list types

        # convert to tensor
        if self.to_tensor:
            img = self.to_tensor(img)
            bboxes = torch.tensor(bboxes)
            category_ids = torch.tensor(category_ids)

        # normalize
        if self.normalize:
            img = self.normalize(img)

        return {"image": img, "bboxes": bboxes, "category_ids": category_ids}
