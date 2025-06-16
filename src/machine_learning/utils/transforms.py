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
        if augmentation is not None:
            if isinstance(augmentation, str):
                self.augmentation = DEFAULT_AUG if augmentation == "default" else ENHANCED_AUG
            else:
                self.augmentation = augmentation

    def __call__(self, data: Sequence[np.ndarray]) -> tuple[torch.Tensor]:
        """应用数据增强"""
        pass


class YoloTransform(CustomTransform):
    def __init__(
        self,
        augmentation: Literal["default", "enhanced"] | A.Compose = None,
        normalize: bool = True,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
        to_tensor: bool = True,
    ):
        """YoloTransform, 实现 albumentations.Compose 到 transforms.Compose 接口的转换.

        Args:
            augmentation (Literal[&quot;default&quot;, &quot;enhanced&quot;] | A.Compose): 增强器.
            normalize (bool, optional): 是否对数据进行归一化处理. Defaults to True.
            mean (Sequence[float] | None, optional): 如果normalize,需要提供均值. Defaults to None.
            std (Sequence[float] | None, optional): 如果normalize,需要提供方差. Defaults to None.
            to_tensor (bool, optional): 是否转化为Tensor. Defaults to True.
        """
        super().__init__(augmentation)

        if normalize:
            self.normalize = transforms.Normalize(mean=mean, std=std)
        if to_tensor:
            self.to_tensor = transforms.ToTensor()  # 只能应用于2/3维

    def __call__(self, data: Sequence[np.ndarray]) -> tuple[torch.Tensor]:
        """应用数据增强

        Args:
            data: 包含 (image, bboxes, category_ids) 的元组

        Returns:
            转换成 Tensor 后的 (image_tensor, bboxes_tensor, category_ids)
        """
        img, bboxes, category_ids = data

        if self.augmentation is not None:
            auged_data = self.augmentation(image=img, bboxes=bboxes, category_ids=category_ids)

            img = auged_data["image"]
            bboxes = auged_data["bboxes"]
            category_ids = auged_data["category_ids"]  # augementation会将其他未注册数据类型转换成列表类型

        # 转化为Tensor
        if self.to_tensor:
            img = self.to_tensor(img)
            bboxes = torch.from_numpy(bboxes)
            category_ids = torch.tensor(category_ids)

        # 归一化
        if self.normalize:
            img = self.normalize(img)

        return img, bboxes, category_ids
