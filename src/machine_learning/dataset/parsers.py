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
from machine_learning.utils import load_cfg, list_from_txt, print_cfg


class ParserBase(ABC):
    """Dataset parser abstract base class."""

    def __init__(self, dataset_cfg: dict[str, Any]) -> None:
        super().__init__()
        self.dataset_cfg = dataset_cfg

        self.dataset_name = self.dataset_cfg["name"]
        self.dataset_path = self.dataset_cfg["path"]

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

    def __init__(self, dataset_cfg: dict[str, Any]):
        super().__init__(dataset_cfg)

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
            "train": {"data": train_imgs, "labels": train_labels},
            "val": {"data": val_imgs, "labels": val_labels},
        }


class CocoParser(ParserBase):
    """
    Coco dataset parser.
    """

    def __init__(self, dataset_cfg: dict[str, dict[str, Any]]):
        super().__init__(dataset_cfg)

        self.train = self.dataset_cfg["train"]
        self.val = self.dataset_cfg["val"]

    def parse(self) -> dict[str, Any]:
        # relative path
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
            "train": {"data": train_imgs, "labels": train_labels},
            "val": {"data": val_imgs, "labels": val_labels},
        }


class FlirAlignedParser(ParserBase):
    r"""Flir_aligned data set parser."""

    def __init__(self, dataset_cfg: dict[str, Any]):
        super().__init__(dataset_cfg)

        self.train = self.dataset_cfg["train"]
        self.val = self.dataset_cfg["val"]

    def parse(self) -> dict[str, Any]:
        # relative paths
        train_irs = list_from_txt(os.path.join(self.dataset_path, self.train))
        val_irs = list_from_txt(os.path.join(self.dataset_path, self.val))

        train_irs = [ir + ".jpeg" for ir in train_irs]
        val_irs = [ir + ".jpeg" for ir in train_irs]

        train_imgs = [f.rsplit("_", maxsplit=1)[0] + "_RGB.jpg" for f in train_irs]
        val_imgs = [f.rsplit("_", maxsplit=1)[0] + "_RGB.jpg" for f in val_irs]

        train_labels = [
            f.replace("JPEGImages", "Annotations", 1).rsplit("_", maxsplit=1)[0] + ".txt" for f in train_irs
        ]
        val_labels = [f.replace("JPEGImages", "Annotations", 1).rsplit("_", maxsplit=1)[0] + ".txt" for f in val_irs]

        # abs path
        train_irs = [os.path.join(self.dataset_path, ir.split("/", 1)[1]) for ir in train_irs]
        val_irs = [os.path.join(self.dataset_path, ir.split("/", 1)[1]) for ir in val_irs]

        train_imgs = [os.path.join(self.dataset_path, img.split("/", 1)[1]) for img in train_imgs]
        val_imgs = [os.path.join(self.dataset_path, img.split("/", 1)[1]) for img in val_imgs]

        train_labels = [os.path.join(self.dataset_path, label.split("/", 1)[1]) for label in train_labels]
        val_labels = [os.path.join(self.dataset_path, label.split("/", 1)[1]) for label in val_labels]

        return {
            "train": {"data": {"imgs": train_imgs, "irs": train_irs}, "labels": train_labels},
            "val": {"data": {"imgs": val_imgs, "irs": val_irs}, "labels": val_labels},
        }


class VedaiParser(ParserBase):
    r"""Multimodal vedai dataset parser."""

    def __init__(self, dataset_cfg: dict[str, Any]):
        super().__init__(dataset_cfg)

        self.train = self.dataset_cfg["train"]
        self.val = self.dataset_cfg["val"]

    def parse(self) -> dict[str, Any]:
        # relative path
        train_dir = os.path.join(self.dataset_path, self.train)
        val_dir = os.path.join(self.dataset_path, self.val)

        train_irs = [f + "_ir.png" for f in train_dir]
        val_irs = [f + "_ir.png" for f in val_ids]

        train_imgs = [f + "_co.png" for f in train_ids]
        val_imgs = [f + "_co.png" for f in val_ids]

        train_labels = [f + ".txt" for f in train_ids]
        val_labels = [f + ".txt" for f in val_ids]

        # abs path
        train_irs = [os.path.join(self.dataset_path, ir) for ir in train_irs]
        val_irs = [os.path.join(self.dataset_path, ir) for ir in val_irs]

        train_imgs = [os.path.join(self.dataset_path, img) for img in train_imgs]
        val_imgs = [os.path.join(self.dataset_path, img) for img in val_imgs]

        train_labels = [os.path.join(self.dataset_path, label) for label in train_labels]
        val_labels = [os.path.join(self.dataset_path, label) for label in val_labels]

        return {
            "train": {"data": {"imgs": train_imgs, "irs": train_irs}, "labels": train_labels},
            "val": {"data": {"imgs": val_imgs, "irs": val_irs}, "labels": val_labels},
        }
