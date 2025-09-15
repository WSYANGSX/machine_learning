"""
Data Parsers, Parse dataset file, return the dataset info.
"""

from __future__ import annotations
from typing import Any

import os
import struct
from abc import ABC, abstractmethod

import numpy as np
from torch.utils.data import Dataset
from machine_learning.utils import load_cfg_from_yaml, list_from_txt, print_cfg


class ParserBase(ABC):
    """Dataset parser abstract base class."""

    def __init__(self, data_cfg: dict[str, Any]) -> None:
        super().__init__()
        self.data_cfg = data_cfg

        self.dataset_name = self.data_cfg["name"]
        self.dataset_path = self.data_cfg["path"]

    @abstractmethod
    def parse(self) -> dict[str, Any]:
        """Parse the train、val and test data or data_path of the dataset.

        Returns:
            dict[str, Any]: train、val and test data or data_path value.
        """
        pass


class MinistParser(ParserBase):
    r"""
    The minist handwritten digit set data parser.
    """

    def __init__(self, data_cfg: dict[str, Any]):
        super().__init__(data_cfg)

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

    def parse(self) -> dict[str, Any]:
        """
        Parse minist dataset metadata
        """
        train_dir = os.path.join(self.dataset_path, "train")
        val_dir = os.path.join(self.dataset_path, "val")

        # data
        train_imgs = self.load_idx3_ubyte(os.path.join(train_dir, "images.idx3-ubyte"))[0]
        val_imgs = self.load_idx3_ubyte(os.path.join(val_dir, "images.idx3-ubyte"))[0]

        # labels
        train_labels = self.load_idx1_ubyte(os.path.join(train_dir, "labels.idx1-ubyte"))[0]
        val_labels = self.load_idx1_ubyte(os.path.join(val_dir, "labels.idx1-ubyte"))[0]

        return {
            "train": [train_imgs, train_labels],
            "val": [val_imgs, val_labels],
        }


class CocoTestParser(ParserBase):
    """
    Coco test dataset parser.
    """

    def __init__(self, data_cfg: dict[str, dict[str, Any]]):
        super().__init__(data_cfg)

        self.train = self.data_cfg["train"]
        self.val = self.data_cfg["val"]

    def parse(self) -> dict[str, Any]:
        # train、 val list
        train_imgs = list_from_txt(os.path.join(self.dataset_path, self.train))
        val_imgs = list_from_txt(os.path.join(self.dataset_path, self.val))

        train_labels = [img.replace("jpg", "txt", 1) for img in train_imgs]
        val_labels = [img.replace("jpg", "txt", 1) for img in val_imgs]

        # abs path
        train_imgs = [os.path.join(self.dataset_path + "/images/train", img) for img in train_imgs]
        val_imgs = [os.path.join(self.dataset_path + "/images/val", img) for img in val_imgs]

        train_labels = [os.path.join(self.dataset_path + "/labels/train", label) for label in train_labels]
        val_labels = [os.path.join(self.dataset_path + "/labels/val", label) for label in val_labels]

        return {
            "train": [train_imgs, train_labels],
            "val": [val_imgs, val_labels],
        }


class CocoParser(ParserBase):
    """
    Coco dataset parser.
    """

    def __init__(self, data_cfg: dict[str, dict[str, Any]]):
        super().__init__(data_cfg)

        self.train = self.data_cfg["train"]
        self.val = self.data_cfg["val"]

    def parse(self) -> dict[str, Any]:
        # train、 val list
        train_imgs = list_from_txt(os.path.join(self.dataset_path, self.train))
        val_imgs = list_from_txt(os.path.join(self.dataset_path, self.val))

        train_labels = [img.replace("images", "labels", 1).replace("jpg", "txt", 1) for img in train_imgs]
        val_labels = [img.replace("images", "labels", 1).replace("jpg", "txt", 1) for img in val_imgs]

        # abs path
        train_imgs = [os.path.join(self.dataset_path, img.split("/", 1)[1]) for img in train_imgs]
        val_imgs = [os.path.join(self.dataset_path, img.split("/", 1)[1]) for img in val_imgs]

        train_labels = [os.path.join(self.dataset_path, label.split("/", 1)[1]) for label in train_labels]
        val_labels = [os.path.join(self.dataset_path, label.split("/", 1)[1]) for label in val_labels]

        return {
            "train": [train_imgs, train_labels],
            "val": [val_imgs, val_labels],
        }


class FlirParser(ParserBase):
    r"""Multimodal Yolo format data set parser."""

    def __init__(self, data_cfg: dict[str, Any]):
        super().__init__(data_cfg)

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


class VedaiParser(ParserBase):
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
