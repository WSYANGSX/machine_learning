from typing import Sequence, Any

import torch
import random
import warnings
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from machine_learning.utils.transforms import TransformBase
from machine_learning.utils.image import resize


class FullDataset(Dataset):
    r"""
    Fully load the dataset.

    It is suitable for small datasets, occupies less memory space and speeds up data reading.
    """

    def __init__(
        self,
        data: np.ndarray | torch.Tensor,
        labels: np.ndarray | torch.Tensor | None = None,
        tansform: TransformBase | None = None,
    ) -> None:
        """
        Initialize the fully load dataset

        Args:
            data (np.ndarray, torch.Tensor): Data
            labels (np.ndarray, torch.Tensor, optional): Labels. Defaults to None.
            tansforms (Compose, BaseTransform, optional): Data converter. Defaults to None.
        """
        super().__init__()

        self.data = data
        self.labels = labels

        self.transform = tansform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        data_sample = self.data[index]

        if self.labels is not None:
            labels_sample = self.labels[index]

        if self.transform:
            data_sample, labels_sample = self.transform(data_sample)

        return data_sample, labels_sample


class LazyDataset(Dataset):
    r"""
    Lazily load dataset.

    It is used for large datasets, reducing memory space occupation, but the data reading speed is relatively slow.
    """

    def __init__(
        self,
        data_paths: Sequence[str],
        label_paths: Sequence[int],
        transform: TransformBase | None = None,
    ):
        """
        Initialize the Lazily load dataset

        Args:
            data_paths (Sequence[str]): Data address list.
            label_paths: (Sequence[int]): Labels address list.
            transform (Compose, BaseTransform, optional): Data converter. Defaults to None.
        """
        super().__init__()

        self.data_paths = data_paths
        self.label_paths = label_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, index) -> Any:
        pass


class YoloDataset(LazyDataset):
    r"""
    Yolo object detection type dataset.

    The object detection dataset is generally large. The inherited lazy loading dataset reduces the memory space
    occupation, but the data reading speed is relatively slow.
    """

    def __init__(
        self,
        img_paths: Sequence[str],
        label_paths: Sequence[int],
        transform: TransformBase = None,
        img_size: int = 416,
        multiscale: bool = False,
        img_size_stride: int | None = 32,
        augment: bool = False,
    ):
        """YoloDataset Inherits from LazyDataset, used for loading the yolo detection data

        Args:
            data_paths (Sequence[str]): Yolo data address list.
            label_paths: (Sequence[int]): Yolo labels address list.
            transform: (YoloTransform): Yolo data converter. Defaults to None.
            img_size: (int): The default required dim of the detected image. Defaults to 416.
            multiscale: (bool): Whether to enable multi-size image training. Defaults to False.
            img_size_stride: (int): The stride of image size change when multi-size image training is enabled. Defaults
            to None.
            augment: (bool): Whether to enable image augment. Defaults to True.
        """
        super().__init__(data_paths=img_paths, label_paths=label_paths, transform=transform)

        self.img_size = img_size
        self.multiscale = multiscale
        self.augment = augment

        if self.multiscale:
            self.img_size_stride = img_size_stride
            self.min_size = self.img_size - 3 * img_size_stride
            self.max_size = self.img_size + 3 * img_size_stride

        self.batch_count = 0

    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, index) -> tuple:
        #  Image
        try:
            img_path = self.data_paths[index % len(self.data_paths)]
            img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        except Exception:
            print(f"Could not read image '{img_path}'.")
            return

        #  Label
        try:
            label_path = self.label_paths[index % len(self.data_paths)]

            # Ignore warning if file is empty
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                labels = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 5)
                bboxes = labels[:, 1:5]
                category_ids = labels[:, 0]

        except Exception:
            print(f"Could not read label '{label_path}'.")
            return

        #  Transform
        if self.transform:
            try:
                transformed_data = self.transform(
                    data={"image": img, "bboxes": bboxes, "category_ids": category_ids}, augment=self.augment
                )
                img = transformed_data["image"]
                bboxes = transformed_data["bboxes"]
                category_ids = transformed_data["category_ids"]

            except Exception:
                print(f"Could not apply transform to image: {img_path}.")
                return

        return img, torch.cat([category_ids.view(-1, 1), bboxes], dim=-1)

    def collate_fn(self, batch) -> tuple:
        self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]

        imgs_ls, cls_bboxes_ls = list(zip(*batch))

        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, self.img_size_stride))

        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs_ls])

        iid_cls_bboxes = torch.cat(
            [
                torch.cat([torch.full((bboxes.shape[0], 1), i, device=bboxes.device), bboxes], dim=-1)
                for i, bboxes in enumerate(cls_bboxes_ls)
            ],
            dim=0,
        )

        return imgs, iid_cls_bboxes
