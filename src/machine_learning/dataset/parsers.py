"""
Data Parsers, Parse dataset file, return the dataset info.
"""

from __future__ import annotations
from typing import Any

import os
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, MISSING

import numpy as np
from torch.utils.data import Dataset
from machine_learning.utils.cfg import BaseCfg
from machine_learning.utils.transforms import TransformBase
from machine_learning.dataset.dataset import YoloDataset, YoloMMDataset, ImgDataset
from machine_learning.utils import load_cfg_from_yaml, list_from_txt, print_cfg


@dataclass
class ParserCfg(BaseCfg):
    """Basic parser configuration"""

    dataset_dir: str = MISSING  # Missing value must be provide when create.
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
        """Parse the information of the data set

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
        metadata = load_cfg_from_yaml(os.path.join(self.dataset_dir, "metadata.yaml"))

        print_cfg("Information of dataset", metadata)

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

        # load data
        train_data = self.load_idx3_ubyte(os.path.join(train_data_dir, "images_train.idx3-ubyte"))[0]
        val_data = self.load_idx3_ubyte(os.path.join(val_data_dir, "images_test.idx3-ubyte"))[0]

        if self.cfg.labels:
            train_labels = self.load_idx1_ubyte(os.path.join(train_data_dir, "labels_train.idx1-ubyte"))[0]
            val_labels = self.load_idx1_ubyte(os.path.join(val_data_dir, "labels_test.idx1-ubyte"))[0]
        else:
            train_labels, val_labels = None, None

        trian_dataset = ImgDataset(train_data, train_labels, self.transforms, augment=False)
        val_dataset = ImgDataset(val_data, val_labels, self.transforms, augment=False)

        return {
            "image_size": metadata["image_size"],
            "train_dataset": trian_dataset,
            "val_dataset": val_dataset,
        }


class YoloParser(ParserBase):
    r"""Yolo format data set parser."""

    def __init__(self, parser_cfg: ParserCfg):
        super().__init__(parser_cfg)

    def parse(self) -> dict[str, Any]:
        metadata = load_cfg_from_yaml(os.path.join(self.dataset_dir, "metadata.yaml"))

        class_names_file = os.path.join(self.dataset_dir, metadata["names_file"])

        print_cfg("Information of dataset", metadata)

        classes = list_from_txt(class_names_file)

        train_img_dir = os.path.join(self.dataset_dir, "images/train2017/")
        train_labels_dir = os.path.join(self.dataset_dir, "annotations/train2017/")

        val_img_dir = os.path.join(self.dataset_dir, "images/val2017/")
        val_labels_dir = os.path.join(self.dataset_dir, "annotations/val2017/")

        # train、 val imgs list
        train_img_ls = list_from_txt(self.dataset_dir + "/train2017.txt")
        val_img_ls = list_from_txt(self.dataset_dir + "/val2017.txt")

        train_img_ls = [img.rsplit("/", 1)[1] for img in train_img_ls]
        val_img_ls = [img.rsplit("/", 1)[1] for img in val_img_ls]

        train_labels_ls = [img.rsplit(".", 1)[0] + ".txt" for img in train_img_ls]
        val_labels_ls = [img.rsplit(".", 1)[0] + ".txt" for img in val_img_ls]

        # abs path
        train_img_paths = [train_img_dir + img for img in train_img_ls]
        val_img_paths = [val_img_dir + img for img in val_img_ls]

        train_labels_paths = [train_labels_dir + label for label in train_labels_ls]
        val_labels_paths = [val_labels_dir + label for label in val_labels_ls]

        return {
            "class_names": classes,
            "calss_nums": len(classes),
            "train_img_paths": train_img_paths,
            "val_img_paths": val_img_paths,
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
            mosaic=True,
        )
        val_dataset = YoloDataset(
            metadata["val_img_paths"],
            metadata["val_labels_paths"],
            transform=self.transforms,
            img_size=self.cfg.img_size,
            multiscale=self.cfg.multiscale,
            img_size_stride=self.cfg.img_size_stride,
            augment=False,
            mosaic=False,
        )

        return {
            "class_names": class_names,
            "class_nums": len(class_names),
            "train_dataset": trian_dataset,
            "val_dataset": val_dataset,
        }


class FlirAlignedParser(ParserBase):
    r"""Multimodal Yolo format data set parser."""

    def __init__(self, parser_cfg: ParserCfg):
        super().__init__(parser_cfg)

    def parse(self) -> dict[str, Any]:
        metadata = load_cfg_from_yaml(os.path.join(self.dataset_dir, "metadata.yaml"))

        print_cfg("Information of dataset", metadata)

        class_names_file = os.path.join(self.dataset_dir, metadata["names_file"])
        img_dir = os.path.join(self.dataset_dir, metadata["images_dir"])
        labels_dir = os.path.join(self.dataset_dir, metadata["labels_dir"])

        classes = list_from_txt(class_names_file)

        # train、 val imgs list
        train_theraml_ls = list_from_txt(os.path.join(self.dataset_dir, metadata["train_ids"]))
        val_theraml_ls = list_from_txt(os.path.join(self.dataset_dir, metadata["val_ids"]))
        train_theraml_ls = [f + ".jpeg" for f in train_theraml_ls]
        val_theraml_ls = [f + ".jpeg" for f in val_theraml_ls]

        train_img_ls = [f.rsplit("_", maxsplit=1)[0] + "_RGB.jpg" for f in train_theraml_ls]
        val_img_ls = [f.rsplit("_", maxsplit=1)[0] + "_RGB.jpg" for f in val_theraml_ls]

        train_labels_ls = [f.rsplit("_", maxsplit=1)[0] + ".txt" for f in train_theraml_ls]
        val_labels_ls = [f.rsplit("_", maxsplit=1)[0] + ".txt" for f in val_theraml_ls]

        # abs path
        train_theraml_paths = [os.path.join(img_dir, theraml) for theraml in train_theraml_ls]
        val_theraml_paths = [os.path.join(img_dir, theraml) for theraml in val_theraml_ls]

        train_img_paths = [os.path.join(img_dir, img) for img in train_img_ls]
        val_img_paths = [os.path.join(img_dir, img) for img in val_img_ls]

        train_labels_paths = [os.path.join(labels_dir, label) for label in train_labels_ls]
        val_labels_paths = [os.path.join(labels_dir, label) for label in val_labels_ls]

        return {
            "class_names": classes,
            "train_img_paths": train_img_paths,
            "val_img_paths": val_img_paths,
            "train_theraml_paths": train_theraml_paths,
            "val_theraml_paths": val_theraml_paths,
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

        trian_dataset = YoloMMDataset(
            img_paths=metadata["train_img_paths"],
            thermal_paths=metadata["train_theraml_paths"],
            label_paths=metadata["train_labels_paths"],
            transform=self.transforms,
            img_size=self.cfg.img_size,
            multiscale=self.cfg.multiscale,
            img_size_stride=self.cfg.img_size_stride,
            augment=True,
            mosaic=True,
        )
        val_dataset = YoloMMDataset(
            metadata["val_img_paths"],
            metadata["val_theraml_paths"],
            metadata["val_labels_paths"],
            transform=self.transforms,
            img_size=self.cfg.img_size,
            multiscale=self.cfg.multiscale,
            img_size_stride=self.cfg.img_size_stride,
            augment=False,
            mosaic=False,
        )

        return {
            "class_names": class_names,
            "class_nums": len(class_names),
            "train_dataset": trian_dataset,
            "val_dataset": val_dataset,
        }


class VEDAIParser(ParserBase):
    r"""Multimodal VEDAI data set parser."""

    def __init__(self, parser_cfg: ParserCfg):
        super().__init__(parser_cfg)

    def parse(self) -> dict[str, Any]:
        metadata = load_cfg_from_yaml(os.path.join(self.dataset_dir, "metadata.yaml"))

        print_cfg("Information of dataset", metadata)

        class_names_file = os.path.join(self.dataset_dir, metadata["names_file"])
        img_dir = os.path.join(self.dataset_dir, metadata["images_dir"])
        labels_dir = os.path.join(self.dataset_dir, metadata["labels_dir"])

        classes = list_from_txt(class_names_file)

        # train、 val imgs list
        train_id_ls = list_from_txt(os.path.join(self.dataset_dir, metadata["train_ids"]))
        val_id_ls = list_from_txt(os.path.join(self.dataset_dir, metadata["val_ids"]))

        train_theraml_ls = [f + "_ir.png" for f in train_id_ls]
        val_theraml_ls = [f + "_ir.png" for f in val_id_ls]

        train_img_ls = [f + "_co.png" for f in train_id_ls]
        val_img_ls = [f + "_co.png" for f in val_id_ls]

        train_labels_ls = [f + ".txt" for f in train_id_ls]
        val_labels_ls = [f + ".txt" for f in val_id_ls]

        # abs path
        train_theraml_paths = [os.path.join(img_dir, theraml) for theraml in train_theraml_ls]
        val_theraml_paths = [os.path.join(img_dir, theraml) for theraml in val_theraml_ls]

        train_img_paths = [os.path.join(img_dir, img) for img in train_img_ls]
        val_img_paths = [os.path.join(img_dir, img) for img in val_img_ls]

        train_labels_paths = [os.path.join(labels_dir, label) for label in train_labels_ls]
        val_labels_paths = [os.path.join(labels_dir, label) for label in val_labels_ls]

        return {
            "class_names": classes,
            "train_img_paths": train_img_paths,
            "val_img_paths": val_img_paths,
            "train_theraml_paths": train_theraml_paths,
            "val_theraml_paths": val_theraml_paths,
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

        trian_dataset = YoloMMDataset(
            img_paths=metadata["train_img_paths"],
            thermal_paths=metadata["train_theraml_paths"],
            label_paths=metadata["train_labels_paths"],
            transform=self.transforms,
            img_size=self.cfg.img_size,
            multiscale=self.cfg.multiscale,
            img_size_stride=self.cfg.img_size_stride,
            augment=True,
            mosaic=True,
        )
        val_dataset = YoloMMDataset(
            metadata["val_img_paths"],
            metadata["val_theraml_paths"],
            metadata["val_labels_paths"],
            transform=self.transforms,
            img_size=self.cfg.img_size,
            multiscale=self.cfg.multiscale,
            img_size_stride=self.cfg.img_size_stride,
            augment=False,
            mosaic=False,
        )

        return {
            "class_names": class_names,
            "class_nums": len(class_names),
            "train_dataset": trian_dataset,
            "val_dataset": val_dataset,
        }
