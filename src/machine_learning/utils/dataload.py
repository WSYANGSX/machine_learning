from __future__ import annotations

import os
import struct
import random
import warnings
from PIL import Image
from copy import deepcopy
from abc import ABC, abstractmethod
from typing import Callable, Sequence, Any
from dataclasses import dataclass, MISSING

import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset

from machine_learning.utils.transforms import CustomTransform, YoloTransform
from machine_learning.utils.image import resize
from machine_learning.utils.others import print_dict, load_config_from_yaml, print_segmentation, list_from_txt


class FullDataset(Dataset):
    r"""
    Fully load the dataset.

    It is suitable for small datasets, occupies less memory space and speeds up data reading.
    """

    def __init__(
        self,
        data: np.ndarray | torch.Tensor,
        labels: np.ndarray | torch.Tensor | None = None,
        tansform: transforms.Compose | CustomTransform | None = None,
    ) -> None:
        """
        Initialize the fully load dataset

        Args:
            data (np.ndarray, torch.Tensor): Data
            labels (np.ndarray, torch.Tensor, optional): Labels. Defaults to None.
            tansforms (Compose, CustomTransform, optional): Data converter. Defaults to None.
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
            data_sample = self.transform(data_sample)

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
        transform: transforms.Compose | CustomTransform | None = None,
    ):
        """
        Initialize the Lazily load dataset

        Args:
            data_paths (Sequence[str]): Data address list.
            label_paths: (Sequence[int]): Labels address list.
            transform (Compose, CustomTransform, optional): Data converter. Defaults to None.
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
        transform: YoloTransform = None,
        img_size: int = 416,
        img_size_stride: int = 32,
        multiscale: bool = False,
    ):
        """YoloDataset Inherits from LazyLoadDataset, used for loading the yolo detection data

        Args:
            data_paths (Sequence[str]): Yolo data address list.
            label_paths: (Sequence[int]): Yolo labels address list.
            transform: (YoloTransform): Yolo data converter. Defaults to None.
            img_size: (int): The default required dim of the detected image. Defaults to 416.
            multiscale: (bool): Whether to enable multi-size image training. Defaults to False.
            img_size_stride: (int): The stride of image size change when multi-size image training is enabled. Defaults to None.
        """
        super().__init__(data_paths=img_paths, label_paths=label_paths, transform=transform)

        self.img_size = img_size
        self.multiscale = multiscale

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

                # Filter out effective bounding boxes
                valid_indices = (bboxes[:, 2] > 0) & (bboxes[:, 3] > 0)
                bboxes = bboxes[valid_indices]
                category_ids = category_ids[valid_indices]

        except Exception:
            print(f"Could not read label '{label_path}'.")
            return

        #  Transform
        if self.transform:
            try:
                img, bboxes, category_ids = self.transform((img, bboxes, category_ids))
            except Exception:
                print(f"Could not apply transform to image: {img_path}.")
                return

        return img, bboxes, category_ids

    def collate_fn(self, batch) -> tuple:
        self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]

        imgs, bboxes, category_ids = list(zip(*batch))
        indices = deepcopy(category_ids)

        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, self.img_size_stride))

        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])

        # Bboxes and category_ids may exist empty tensors.
        bboxes = torch.cat(bboxes, 0)
        category_ids = torch.cat(category_ids, 0)

        for i, index in enumerate(indices):
            index[:] = i
        indices = torch.cat(indices, 0)

        return imgs, bboxes, category_ids, indices


class ParserFactory:
    r"""The factory class is used to generate specific data parsers and follows the open-closed principle."""

    _parser_registry: dict[str, DatasetParser] = {}

    def __init__(self):
        pass

    @property
    def parsers(self) -> list[str]:
        return list(self._parser_registry.keys())

    @classmethod
    def register_parser(cls, dataset_type: str) -> Callable:
        def parser_wrapper(parser_cls: DatasetParser) -> None:
            cls._parser_registry[dataset_type] = parser_cls
            print(f"DataLoaderFactory has registred dataset_parser '{parser_cls.__name__}'.")
            return parser_cls

        return parser_wrapper

    def parser_create(self, parser_cfg: ParserCfg) -> DatasetParser:
        dataset_dir = os.path.abspath(parser_cfg.dataset_dir)
        metadata = self._load_metadata(dataset_dir)

        dataset_type: str = metadata["dataset_type"]
        # Dynamically obtain the parser
        if dataset_type not in self._parser_registry:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
        parser_cls = self._parser_registry[dataset_type]

        return parser_cls(parser_cfg)

    def _load_metadata(self, dataset_dir: str) -> dict:
        "Load the metadata file"
        metadata_path = os.path.join(dataset_dir, "metadata.yaml")
        return load_config_from_yaml(metadata_path)

    def __str__(self):
        return f"DataLoaderFactory(parsers={self.parsers})"


@dataclass
class ParserCfg:
    dataset_dir: str = MISSING
    labels: bool = MISSING
    transforms: transforms.Compose | CustomTransform | None = None


class DatasetParser(ABC):
    """Dataset parser abstract base class."""

    def __init__(self, parser_cfg: ParserCfg) -> None:
        super().__init__()
        self.cfg = parser_cfg

        self.dataset_dir = self.cfg.dataset_dir
        self.labels = self.cfg.labels
        self.transforms = self.cfg.transforms

    @abstractmethod
    def parse(self) -> dict[str, Any]:
        """Parse the metadata information of the data set

        Returns:
            dict[str, Any]: Meta-information return value.
        """
        pass

    @abstractmethod
    def create(self, *args, **kwargs) -> dict[str, Dataset]:
        """Create a dataset based on the parsed data information of the dataset.

        Returns:
            dict[str, Dataset]: Return the dictionary containing the training (train_dataset), validation (val_dataset)
            datasets and special meta info of dataset.
        """
        pass

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"dataset_dir={self.dataset_dir}, "
            f"labels={self.labels}, "
            f"transforms={self.transforms})"
        )


@ParserFactory.register_parser("minist")
class MinistParser(DatasetParser):
    r"""
    The minist handwritten digit set data parser, due to the small volume of misit data sets, adopts a full data loading
    method.
    """

    def __init__(self, parser_cfg: ParserCfg):
        super().__init__(parser_cfg)

    def parse(self) -> dict[str, Any]:
        """
        Parse minist dataset metadata
        """
        metadata = load_config_from_yaml(os.path.join(self.dataset_dir, "metadata.yaml"))

        dataset_name = metadata["dataset_name"]
        if metadata["dataset_type"] != "minist":
            raise TypeError(f"Dataset {dataset_name} is not the type of minist.")

        print_segmentation()
        print("Information of dataset:")
        print_dict(metadata)
        print_segmentation()

        return metadata

    @staticmethod
    def load_idx3_ubyte(dataset_dir: str) -> tuple:
        with open(dataset_dir, "rb") as f:
            magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
            images = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows, cols)

            return images, magic, num_images, rows, cols

    @staticmethod
    def load_idx1_ubyte(dataset_dir: str) -> tuple:
        with open(dataset_dir, "rb") as f:
            magic, num_labels = struct.unpack(">II", f.read(8))
            labels = np.fromfile(f, dtype=np.uint8)

            return labels, magic, num_labels

    def create(self) -> dict[str, Dataset]:
        """Create data based on the configuration information of MinistParser

        Returns:
            dict[str, Dataset]: Return the dictionary containing the training (train) and validation (val) datasets.
        """
        self.parse()

        train_data_dir = os.path.join(self.dataset_dir, "train")
        val_data_dir = os.path.join(self.dataset_dir, "test")
        print("[INFO] Train data directory path: ", train_data_dir)
        print("[INFO] Val data directory path: ", val_data_dir)

        # load data
        train_data = self.load_idx3_ubyte(os.path.join(train_data_dir, "images_train.idx3-ubyte"))[0]
        val_data = self.load_idx3_ubyte(os.path.join(val_data_dir, "images_test.idx3-ubyte"))[0]

        if self.cfg.labels:
            train_labels = self.load_idx1_ubyte(os.path.join(train_data_dir, "labels_train.idx1-ubyte"))[0]
            val_labels = self.load_idx1_ubyte(os.path.join(val_data_dir, "labels_test.idx1-ubyte"))[0]
        else:
            train_labels, val_labels = None, None

        trian_dataset = FullDataset(train_data, train_labels, self.transforms)
        val_dataset = FullDataset(val_data, val_labels, self.transforms)

        return {"train_dataset": trian_dataset, "val_dataset": val_dataset}


@ParserFactory.register_parser("yolo")
class YoloParser(DatasetParser):
    r"""Yolo format data set parser."""

    def __init__(self, parser_cfg: ParserCfg):
        super().__init__(parser_cfg)

    def parse(self) -> dict[str, Any]:
        metadata = load_config_from_yaml(os.path.join(self.dataset_dir, "metadata.yaml"))

        dataset_name = metadata["dataset_name"]
        if metadata["dataset_type"] != "yolo":
            raise TypeError(f"Dataset {dataset_name} is not the type of yolo.")

        class_names_file = os.path.join(self.dataset_dir, metadata["names_file"])

        print_segmentation()
        print("Information of dataset:")
        print_dict(metadata)
        print_segmentation()

        classes = list_from_txt(class_names_file)

        train_img_dir = os.path.join(self.dataset_dir, "images/val/")
        train_labels_dir = os.path.join(self.dataset_dir, "labels/val/")

        val_img_dir = os.path.join(self.dataset_dir, "images/val/")
        val_labels_dir = os.path.join(self.dataset_dir, "labels/val/")

        # train、 val imgs list
        train_img_ls = list_from_txt(self.dataset_dir + "/images_val.txt")
        val_img_ls = list_from_txt(self.dataset_dir + "/images_val.txt")

        train_labels_ls = [img.rsplit(".", 1)[0] + ".txt" for img in train_img_ls]
        val_labels_ls = [img.rsplit(".", 1)[0] + ".txt" for img in val_img_ls]

        # abs path
        train_img_paths = [train_img_dir + img for img in train_img_ls]
        val_img_paths = [val_img_dir + img for img in val_img_ls]
        train_labels_paths = [train_labels_dir + label for label in train_labels_ls]
        val_labels_paths = [val_labels_dir + label for label in val_labels_ls]

        return {
            "class_names": classes,
            "class_nums": len(classes),
            "train_img_paths": train_img_paths,
            "val_img_paths": val_img_paths,
            "train_labels_paths": train_labels_paths,
            "val_labels_paths": val_labels_paths,
        }

    def create(self, img_size: int = 416, img_size_stride: int = 32, multiscale: bool = False) -> dict[str, Dataset]:
        """Create a dataset based on the configuration information of YoloParser.

        Returns:
            dict[str, Dataset]: Return the dictionary containing the training (train) and validation (val) datasets.
        """
        # 解析类别和路径信息
        metadata = self.parse()
        class_names = metadata["class_names"]
        class_nums = metadata["class_nums"]

        trian_dataset = YoloDataset(
            img_paths=metadata["train_img_paths"],
            label_paths=metadata["train_labels_paths"],
            transform=self.transforms,
            img_size=img_size,
            img_size_stride=img_size_stride,
            multiscale=multiscale,
        )
        val_dataset = YoloDataset(
            metadata["val_img_paths"],
            metadata["val_labels_paths"],
            self.transforms,
            transform=self.transforms,
            img_size=img_size,
            img_size_stride=img_size_stride,
            multiscale=multiscale,
        )

        return {
            "class_nums": class_nums,
            "class_names": class_names,
            "train_dataset": trian_dataset,
            "val_dataset": val_dataset,
        }
