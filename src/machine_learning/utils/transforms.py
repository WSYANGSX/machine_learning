import torch
import numpy as np
import albumentations as A
from torchvision import transforms

from typing import Literal, Sequence

from machine_learning.utils.augmentations import DEFAULT_AUG, ENHANCED_AUG


class CustomTransform:
    r"""自定义Transform基类, 实现对 albumentations.Compose 和 transforms.Compose 接口的封装."""

    def __init__(
        self,
        augmentation: Literal["default", "enhanced"] | A.Compose | None = None,
    ):
        if not augmentation:
            if isinstance(augmentation, str):
                self.augmentation = DEFAULT_AUG if augmentation == "default" else ENHANCED_AUG
            else:
                self.augmentation = augmentation

    def __call__(self, data: Sequence[np.ndarray]) -> tuple[torch.Tensor]:
        """应用数据增强"""
        pass


class YoloTransform(CustomTransform):
    r"""YoloTransform, 实现 albumentations.Compose 到 transforms.Compose 接口的转换."""

    def __init__(
        self,
        augmentation: Literal["default", "enhanced"] | A.Compose,
        mean: Sequence[float],
        std: Sequence[float],
    ):
        super().__init__(augmentation)

        self.to_tensor = transforms.ToTensor()  # 只能应用于2/3维
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, data: Sequence[np.ndarray]) -> tuple[torch.Tensor]:
        """应用数据增强

        Args:
            data: 包含 (image, bboxes, category_ids) 的元组

        Returns:
            转换成 Tensor 后的 (image_tensor, bboxes_tensor, category_ids)
        """
        img, bboxes, category_ids = data
        aug_data = self.augmentation(image=img, bboxes=bboxes, category_ids=category_ids)

        img = aug_data["image"]
        bboxes = aug_data["bboxes"]
        category_ids = aug_data["category_ids"]

        # 转化为Tensor
        img = self.to_tensor(img)
        bboxes = self.to_tensor(bboxes)
        category_ids = torch.from_numpy(category_ids)

        # 归一化
        img = self.normalize(img)

        return img, bboxes, category_ids
