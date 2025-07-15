from __future__ import annotations
from typing import Any

import os
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, MISSING

import numpy as np
from torch.utils.data import Dataset
from machine_learning.utils.transforms import TransformBase
from machine_learning.utils.dataset import ImgEagerDataset, YoloDataset
from machine_learning.utils.others import print_dict, load_config_from_yaml, print_segmentation, list_from_txt


@dataclass
class ParserCfg:
    """Basic parser configuration"""

    dataset_dir: str = MISSING
    labels: bool = MISSING
    tfs: TransformBase | None = None


@dataclass
class YoloParserCfg(ParserCfg):
    """YOLO parser configuration"""

    img_size: int = 416
    multiscale: bool = False
    img_size_stride: int | None = 32


class ParserBase(ABC):
    """Dataset parser abstract base class."""

    def __init__(self, parser_cfg: ParserCfg) -> None:
        super().__init__()
        self.cfg = parser_cfg

        self.dataset_dir = self.cfg.dataset_dir
        self.labels = self.cfg.labels
        self.transforms = self.cfg.tfs

    @abstractmethod
    def parse(self) -> dict[str, Any]:
        """Parse the metadata information of the data set

        Returns:
            dict[str, Any]: Meta-information return value.
        """
        pass

    @abstractmethod
    def create(self) -> dict[str, Dataset]:
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


class MinistParser(ParserBase):
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
        metadata = self.parse()

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

        trian_dataset = ImgEagerDataset(train_data, train_labels, self.transforms, augment=False)
        val_dataset = ImgEagerDataset(val_data, val_labels, self.transforms, augment=False)

        return {
            "image_shape": metadata["image_shape"],
            "train_dataset": trian_dataset,
            "val_dataset": val_dataset,
        }


class YoloParser(ParserBase):
    r"""Yolo format data set parser."""

    def __init__(self, parser_cfg: ParserCfg):
        super().__init__(parser_cfg)

    def parse(self) -> dict[str, Any]:
        metadata = load_config_from_yaml(os.path.join(self.dataset_dir, "metadata.yaml"))

        dataset_name = metadata["name"]
        if metadata["data_type"] != "yolo":
            raise TypeError(f"Dataset {dataset_name} is not the type of yolo.")

        class_names_file = os.path.join(self.dataset_dir, metadata["names_file"])

        print_segmentation()
        print("Information of dataset:")
        print_dict(metadata)
        print_segmentation()

        classes = list_from_txt(class_names_file)

        train_img_dir = os.path.join(self.dataset_dir, "images/train/")
        train_labels_dir = os.path.join(self.dataset_dir, "labels/train/")

        val_img_dir = os.path.join(self.dataset_dir, "images/val/")
        val_labels_dir = os.path.join(self.dataset_dir, "labels/val/")

        test_img_dir = os.path.join(self.dataset_dir, "images/test/")

        # train、 val imgs list
        train_img_ls = list_from_txt(self.dataset_dir + "/images_train.txt")
        val_img_ls = list_from_txt(self.dataset_dir + "/images_val.txt")
        test_img_ls = list_from_txt(self.dataset_dir + "/images_test.txt")

        train_labels_ls = [img.rsplit(".", 1)[0] + ".txt" for img in train_img_ls]
        val_labels_ls = [img.rsplit(".", 1)[0] + ".txt" for img in val_img_ls]

        # abs path
        train_img_paths = [train_img_dir + img for img in train_img_ls]
        val_img_paths = [val_img_dir + img for img in val_img_ls]
        test_img_paths = [test_img_dir + img for img in test_img_ls]
        train_labels_paths = [train_labels_dir + label for label in train_labels_ls]
        val_labels_paths = [val_labels_dir + label for label in val_labels_ls]

        return {
            "class_names": classes,
            "calss_nums": len(classes),
            "train_img_paths": train_img_paths,
            "val_img_paths": val_img_paths,
            "test_img_paths": test_img_paths,
            "train_labels_paths": train_labels_paths,
            "val_labels_paths": val_labels_paths,
        }

    def create(self) -> dict[str, Dataset]:
        """Create a dataset based on the configuration information of YoloParser.

        Returns:
            dict[str, Dataset]: Return the dictionary containing the training (train) and validation (val) datasets.
        """
        # 解析类别和路径信息
        metadata = self.parse()
        class_names = metadata["class_names"]

        trian_dataset = YoloDataset(
            img_paths=metadata["train_img_paths"],
            label_paths=metadata["train_labels_paths"],
            transform=self.transforms,
            img_size=self.cfg.img_size,
            multiscale=self.cfg.multiscale,
            img_size_stride=self.cfg.img_size_stride,
            augment=True,
        )
        val_dataset = YoloDataset(
            metadata["val_img_paths"],
            metadata["val_labels_paths"],
            transform=self.transforms,
            img_size=self.cfg.img_size,
            multiscale=self.cfg.multiscale,
            img_size_stride=self.cfg.img_size_stride,
            augment=True,
        )
        test_dataset = YoloDataset(
            img_paths=metadata["test_img_paths"],
            label_paths=None,
            transform=self.transforms,
            augment=False,
        )

        return {
            "class_names": class_names,
            "class_nums": len(class_names),
            "train_dataset": trian_dataset,
            "val_dataset": val_dataset,
            "test_dataset": test_dataset,
        }
