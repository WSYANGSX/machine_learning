from typing import Sequence, Any
import random
import warnings
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A

from machine_learning.utils.transforms import TransformBase, ImgTransform
from machine_learning.utils.image import resize


class SimpleDataset(Dataset):
    r"""
    Simple full Load the dataset of data.

    It is suitable for small datasets, occupies less memory space and speeds up data reading.
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray | None = None,
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
        pass


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


class ImgDataset(SimpleDataset):
    r"""
    Fully load the dataset of image.

    It is suitable for small datasets, occupies less memory space and speeds up data reading.
    """

    def __init__(
        self,
        imgs: np.ndarray | torch.Tensor,
        labels: np.ndarray | torch.Tensor | None = None,
        tansform: ImgTransform | None = None,
        augment: bool = False,
    ) -> None:
        """
        Initialize the fully load dataset

        Args:
            data (np.ndarray, torch.Tensor): Data
            labels (np.ndarray, torch.Tensor, optional): Labels. Defaults to None.
            tansforms (Compose, BaseTransform, optional): Data converter. Defaults to None.
        """
        super().__init__(data=imgs, labels=labels, tansform=tansform)

        self.augment = augment
        self.transform = tansform

    @property
    def imgs(self):
        return self.data

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, index):
        img = self.imgs[index]

        if self.labels is not None:
            label = self.labels[index]
            if self.transform:
                transformed_data = self.transform(data={"image": img}, augment=self.augment)
                img = transformed_data["image"]
            label = torch.tensor(label)

            return img, label
        else:
            if self.transform:
                transformed_data = self.transform(data={"image": img}, augment=self.augment)
                img = transformed_data["image"]

            return img


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
        transform: ImgTransform = None,
        img_size: int = 640,
        multiscale: bool = False,
        img_size_stride: int | None = 32,
        augment: bool = False,
        mosaic: bool = False,
    ):
        """YoloDataset Inherits from LazyDataset, used for loading the yolo detection data

        Args:
            data_paths (Sequence[str]): Yolo data address list.
            label_paths: (Sequence[int]): Yolo labels address list.
            transform: (YoloTransform): Yolo data converter. Defaults to None.
            img_size: (int): The default required dim of the detected image. Defaults to 640.
            multiscale: (bool): Whether to enable multi-size image training. Defaults to False.
            img_size_stride: (int): The stride of image size change when multi-size image training is enabled. Defaults
            to None.
            augment: (bool): Whether to enable image augment. Defaults to True.
        """
        super().__init__(data_paths=img_paths, label_paths=label_paths, transform=transform)

        self.img_size = img_size
        self.multiscale = multiscale
        self.augment = augment
        self.mosaic = mosaic

        if self.multiscale:
            self.img_size_stride = img_size_stride
            self.min_size = self.img_size - 3 * img_size_stride
            self.max_size = self.img_size + 3 * img_size_stride

        self.batch_count = 0

        self.buffer = []
        self.max_buffer_length = 1000

    def __len__(self) -> int:
        return len(self.data_paths)

    def load_data(self, index) -> tuple:
        #  Image
        try:
            img_path = self.data_paths[index % len(self.data_paths)]
            img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        except Exception:
            print(f"Could not read image '{img_path}'.")
            return None

        #  Label
        try:
            label_path = self.label_paths[index % len(self.data_paths)]

            # Ignore warning if file is empty
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                labels = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 5)

                if labels.size > 0:
                    labels = labels.reshape(-1, 5)
                    valid_mask = (labels[:, 3] > 0) & (labels[:, 4] > 0)
                    labels = labels[valid_mask]

                bboxes = labels[:, 1:5]
                category_ids = labels[:, 0]

        except Exception:
            print(f"Could not read label '{label_path}'.")
            return None

        return img, bboxes, category_ids, img_path

    def __getitem__(self, index) -> tuple:
        img0, bboxes0, category_ids0, img_path0 = self.load_data(index=index)

        # mosaic
        if self.mosaic:
            if len(self.buffer) > self.max_buffer_length:
                self.buffer.pop(0)
            self.buffer.append(index)

            mosaic_transform = A.Compose(
                [
                    A.Mosaic(
                        grid_yx=(2, 2),
                        cell_shape=(320, 320),
                        fit_mode="contain",
                        target_size=(640, 640),
                        metadata_key="mosaic_metadata",
                        p=0.8,
                    ),
                ],
                bbox_params=A.BboxParams(
                    format="yolo",
                    label_fields=["category_ids"],
                    min_visibility=0.1,
                    min_height=0.01,
                    min_width=0.01,
                    clip=True,
                ),
                p=1,
            )

            mosaic_data = []
            for _ in range(3):
                random_index = random.choice(self.buffer)
                img, bboxes, category_ids, _ = self.load_data(index=random_index)
                mosaic_data.append(
                    {
                        "image": img,
                        "bboxes": bboxes,
                        "category_ids": category_ids,
                    }
                )

            primary_data = {
                "image": img0,
                "bboxes": bboxes0,
                "category_ids": category_ids0,
                "mosaic_metadata": mosaic_data[:],
            }
            result = mosaic_transform(**primary_data)

            img = result["image"]
            bboxes = result["bboxes"]
            category_ids = result["category_ids"]

        #  Transform
        if self.transform:
            try:
                transformed_data = self.transform(
                    data={"image": img0, "bboxes": bboxes0, "category_ids": category_ids0}, augment=self.augment
                )
                img = transformed_data["image"]
                bboxes = torch.tensor(transformed_data["bboxes"])
                category_ids = torch.tensor(transformed_data["category_ids"])

            except Exception as e:
                print(f"Could not apply transform to image: {img_path0}. {e}")
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


class YoloMMDataset(LazyDataset):
    r"""
    Yolo multimodal object detection type dataset.

    The object detection dataset is generally large. The inherited lazy loading dataset reduces the memory space
    occupation, but the data reading speed is relatively slow.
    """

    def __init__(
        self,
        img_paths: Sequence[str],
        thermal_paths: Sequence[str],
        label_paths: Sequence[int],
        transform: ImgTransform = None,
        img_size: int = 640,
        multiscale: bool = False,
        img_size_stride: int | None = 32,
        augment: bool = False,
        mosaic: bool = False,
    ):
        """YoloMM dataset Inherits from LazyDataset, used for loading the yolo detection data

        Args:
            img_paths (Sequence[str]): imgs data address list.
            thermal_paths (Sequence[str]): thermal data address list.
            label_paths: (Sequence[int]): Labels address list.
            transform: (YoloTransform): Yolo data converter. Defaults to None.
            img_size: (int): The default required dim of the detected image. Defaults to 416.
            multiscale: (bool): Whether to enable multi-size image training. Defaults to False.
            img_size_stride: (int): The stride of image size change when multi-size image training is enabled. Defaults
            to None.
            augment: (bool): Whether to enable image augment. Defaults to False.
            mosaic: (bool): Whether to enable image mosaic. Defaults to False.
        """
        super().__init__(data_paths=img_paths, label_paths=label_paths, transform=transform)

        self.thermal_paths = thermal_paths

        self.img_size = img_size
        self.multiscale = multiscale
        self.augment = augment
        self.mosaic = mosaic

        if self.multiscale:
            self.img_size_stride = img_size_stride
            self.min_size = self.img_size - 3 * img_size_stride
            self.max_size = self.img_size + 3 * img_size_stride

        self.batch_count = 0

        self.buffer = []
        self.max_buffer_length = 100

    def __len__(self) -> int:
        return len(self.data_paths)

    def load_data(self, index) -> tuple:
        #  Image
        try:
            img_path = self.data_paths[index % len(self.data_paths)]
            img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        except Exception:
            print(f"Could not read image '{img_path}'.")
            return None

        #  Thremal
        try:
            thermal_path = self.thermal_paths[index % len(self.data_paths)]
            thermal = np.array(Image.open(thermal_path).convert("L"), dtype=np.uint8)
        except Exception:
            print(f"Could not read thermal image '{thermal_path}'.")
            return None

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
            return None

        return img, thermal, bboxes, category_ids, img_path

    def __getitem__(self, index) -> tuple:
        img, thermal, bboxes, category_ids, img_path = self.load_data(index=index)

        # mosaic
        if self.mosaic:
            if len(self.buffer) > self.max_buffer_length:
                self.buffer.pop(0)
            self.buffer.append(index)

            mosaic_transform = A.Compose(
                [
                    A.Mosaic(
                        grid_yx=(2, 2),
                        cell_shape=(320, 320),
                        fit_mode="contain",
                        target_size=(640, 640),
                        metadata_key="mosaic_metadata",
                        p=0.6,
                    ),
                ],
                bbox_params=A.BboxParams(
                    format="yolo",
                    label_fields=["category_ids"],
                    min_visibility=0.1,
                    min_height=0.01,
                    min_width=0.01,
                    clip=True,
                ),
                p=1,
            )

            mosaic_data = []
            for _ in range(3):
                random_index = random.choice(self.buffer)
                img, thermal, bboxes, category_ids, _ = self.load_data(index=random_index)
                mosaic_data.append(
                    {
                        "image": img,
                        "mask": thermal,
                        "bboxes": bboxes,
                        "category_ids": category_ids,
                    }
                )

            primary_data = {
                "image": img,
                "mask": thermal,
                "bboxes": bboxes,
                "category_ids": category_ids,
                "mosaic_metadata": mosaic_data[:],
            }
            result = mosaic_transform(**primary_data)

            img = result["image"]
            thermal = result["mask"]
            bboxes = result["bboxes"]
            category_ids = result["category_ids"]

        #  Transform
        if self.transform:
            try:
                transformed_data = self.transform(
                    data={"image": img, "thermal": thermal, "bboxes": bboxes, "category_ids": category_ids},
                    augment=self.augment,
                )
                img = transformed_data["image"]
                thermal = transformed_data["thermal"]
                bboxes = torch.tensor(transformed_data["bboxes"])
                category_ids = torch.tensor(transformed_data["category_ids"])

            except Exception:
                print(f"Could not apply transform to image: {img_path}.")
                return

        return img, thermal, torch.cat([category_ids.view(-1, 1), bboxes], dim=-1)

    def collate_fn(self, batch) -> tuple:
        self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]

        imgs_ls, thermal_ls, cls_bboxes_ls = list(zip(*batch))

        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, self.img_size_stride))

        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs_ls])
        thermals = torch.stack([resize(thermal, self.img_size) for thermal in thermal_ls])

        iid_cls_bboxes = torch.cat(
            [
                torch.cat([torch.full((bboxes.shape[0], 1), i, device=bboxes.device), bboxes], dim=-1)
                for i, bboxes in enumerate(cls_bboxes_ls)
            ],
            dim=0,
        )

        return imgs, thermals, iid_cls_bboxes
