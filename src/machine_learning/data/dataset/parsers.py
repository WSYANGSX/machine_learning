"""
Data Parsers, Parse dataset file, return the dataset info.
"""

from __future__ import annotations
from typing import Any

import os
import struct
from abc import ABC, abstractmethod

import numpy as np
from machine_learning.utils import list_from_txt


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
            "train": {"imgs": train_imgs, "labels": train_labels},
            "val": {"imgs": val_imgs, "labels": val_labels},
        }


class FlirAlignedParser(ParserBase):
    r"""Flir_aligned data set parser."""

    def __init__(self, dataset_cfg: dict[str, Any]):
        super().__init__(dataset_cfg)

        self.train_imgs_dir = self.dataset_cfg["train_imgs_dir"]
        self.train_irs_dir = self.dataset_cfg["train_irs_dir"]
        self.val_imgs_dir = self.dataset_cfg["val_imgs_dir"]
        self.val_irs_dir = self.dataset_cfg["val_irs_dir"]

        self.train_labels_dir = self.dataset_cfg["train_labels_dir"]
        self.val_labels_dir = self.dataset_cfg["val_labels_dir"]

        self.train_ids = extract_ids_from_dir(os.path.join(self.dataset_path, self.train_imgs_dir))
        self.val_ids = extract_ids_from_dir(os.path.join(self.dataset_path, self.val_imgs_dir))

    def parse(self) -> dict[str, Any]:
        # train path
        train_imgs = [os.path.join(self.dataset_path, self.train_imgs_dir) + f"/{id}.jpg" for id in self.train_ids]
        train_irs = [os.path.join(self.dataset_path, self.train_irs_dir) + f"/{id}.jpeg" for id in self.train_ids]
        train_labels = [os.path.join(self.dataset_path, self.train_labels_dir) + f"/{id}.txt" for id in self.train_ids]

        # val path
        val_imgs = [os.path.join(self.dataset_path, self.val_imgs_dir) + f"/{id}.jpg" for id in self.val_ids]
        val_irs = [os.path.join(self.dataset_path, self.val_irs_dir) + f"/{id}.jpeg" for id in self.val_ids]
        val_labels = [os.path.join(self.dataset_path, self.val_labels_dir) + f"/{id}.txt" for id in self.val_ids]

        return {
            "train": {"data": {"imgs": train_imgs, "irs": train_irs}, "labels": train_labels},
            "val": {"data": {"imgs": val_imgs, "irs": val_irs}, "labels": val_labels},
        }


class VedaiParser(ParserBase):
    r"""Multimodal vedai dataset parser."""

    def __init__(self, dataset_cfg: dict[str, Any]):
        super().__init__(dataset_cfg)

        self.train_dir = os.path.join(self.dataset_path, self.dataset_cfg["train_dir"])
        self.val_dir = os.path.join(self.dataset_path, self.dataset_cfg["val_dir"])
        self.train_ids = list_from_txt(os.path.join(self.dataset_path, self.dataset_cfg["train_ids"]))
        self.val_ids = list_from_txt(os.path.join(self.dataset_path, self.dataset_cfg["val_ids"]))

    def parse(self) -> dict[str, Any]:
        # trian paths
        train_imgs = [os.path.join(self.train_dir, "images") + f"/{id}.png" for id in self.train_ids]
        train_irs = [os.path.join(self.train_dir, "irs") + f"/{id}.png" for id in self.train_ids]
        train_labels = [os.path.join(self.train_dir, "labels") + f"/{id}.txt" for id in self.train_ids]
        # label paths
        val_imgs = [os.path.join(self.val_dir, "images") + f"/{id}.png" for id in self.val_ids]
        val_irs = [os.path.join(self.val_dir, "irs") + f"/{id}.png" for id in self.val_ids]
        val_labels = [os.path.join(self.val_dir, "labels") + f"/{id}.txt" for id in self.val_ids]

        # Multi modal names should be uniformly in the singular form for convenience
        return {
            "train": {"data": {"imgs": train_imgs, "irs": train_irs}, "labels": train_labels},
            "val": {"data": {"imgs": val_imgs, "irs": val_irs}, "labels": val_labels},
        }


class DVParser(ParserBase):
    r"""Drone Vehicle dataset parser."""

    def __init__(self, dataset_cfg: dict[str, Any]):
        super().__init__(dataset_cfg)

        self.train_imgs_dir = self.dataset_cfg["train_imgs_dir"]
        self.train_irs_dir = self.dataset_cfg["train_irs_dir"]
        self.val_imgs_dir = self.dataset_cfg["val_imgs_dir"]
        self.val_irs_dir = self.dataset_cfg["val_irs_dir"]
        self.test_imgs_dir = self.dataset_cfg["test_imgs_dir"]
        self.test_irs_dir = self.dataset_cfg["test_irs_dir"]

        self.train_labels_dir = self.dataset_cfg["train_labels_dir"]
        self.val_labels_dir = self.dataset_cfg["val_labels_dir"]
        self.test_labels_dir = self.dataset_cfg["test_labels_dir"]

        self.train_ids = extract_ids_from_dir(os.path.join(self.dataset_path, self.train_imgs_dir))
        self.val_ids = extract_ids_from_dir(os.path.join(self.dataset_path, self.val_imgs_dir))
        self.test_ids = extract_ids_from_dir(os.path.join(self.dataset_path, self.test_imgs_dir))

    def parse(self) -> dict[str, Any]:
        # train path
        train_imgs = [os.path.join(self.dataset_path, self.train_imgs_dir) + f"/{id}.jpg" for id in self.train_ids]
        train_irs = [os.path.join(self.dataset_path, self.train_irs_dir) + f"/{id}.jpg" for id in self.train_ids]
        train_labels = [os.path.join(self.dataset_path, self.train_labels_dir) + f"/{id}.txt" for id in self.train_ids]

        # val path
        val_imgs = [os.path.join(self.dataset_path, self.val_imgs_dir) + f"/{id}.jpg" for id in self.val_ids]
        val_irs = [os.path.join(self.dataset_path, self.val_irs_dir) + f"/{id}.jpg" for id in self.val_ids]
        val_labels = [os.path.join(self.dataset_path, self.val_labels_dir) + f"/{id}.txt" for id in self.val_ids]

        # test path
        test_imgs = [os.path.join(self.dataset_path, self.test_imgs_dir) + f"/{id}.jpg" for id in self.test_ids]
        test_irs = [os.path.join(self.dataset_path, self.test_irs_dir) + f"/{id}.jpg" for id in self.test_ids]
        test_labels = [os.path.join(self.dataset_path, self.test_labels_dir) + f"/{id}.txt" for id in self.test_ids]

        return {
            "train": {"data": {"imgs": train_imgs, "irs": train_irs}, "labels": train_labels},
            "val": {"data": {"imgs": val_imgs, "irs": val_irs}, "labels": val_labels},
            "test": {"data": {"imgs": test_imgs, "irs": test_irs}, "labels": test_labels},
        }


class LLVIPParser(ParserBase):
    r"""LLVIP data set parser."""

    def __init__(self, dataset_cfg: dict[str, Any]):
        super().__init__(dataset_cfg)

        self.train_imgs_dir = self.dataset_cfg["train_imgs_dir"]
        self.train_irs_dir = self.dataset_cfg["train_irs_dir"]
        self.val_imgs_dir = self.dataset_cfg["val_imgs_dir"]
        self.val_irs_dir = self.dataset_cfg["val_irs_dir"]

        self.train_labels_dir = self.dataset_cfg["train_labels_dir"]
        self.val_labels_dir = self.dataset_cfg["val_labels_dir"]

        self.train_ids = extract_ids_from_dir(os.path.join(self.dataset_path, self.train_imgs_dir))
        self.val_ids = extract_ids_from_dir(os.path.join(self.dataset_path, self.val_imgs_dir))

    def parse(self) -> dict[str, Any]:
        # train path
        train_imgs = [os.path.join(self.dataset_path, self.train_imgs_dir) + f"/{id}.jpg" for id in self.train_ids]
        train_irs = [os.path.join(self.dataset_path, self.train_irs_dir) + f"/{id}.jpg" for id in self.train_ids]
        train_labels = [os.path.join(self.dataset_path, self.train_labels_dir) + f"/{id}.txt" for id in self.train_ids]

        # val path
        val_imgs = [os.path.join(self.dataset_path, self.val_imgs_dir) + f"/{id}.jpg" for id in self.val_ids]
        val_irs = [os.path.join(self.dataset_path, self.val_irs_dir) + f"/{id}.jpg" for id in self.val_ids]
        val_labels = [os.path.join(self.dataset_path, self.val_labels_dir) + f"/{id}.txt" for id in self.val_ids]

        return {
            "train": {"data": {"imgs": train_imgs, "irs": train_irs}, "labels": train_labels},
            "val": {"data": {"imgs": val_imgs, "irs": val_irs}, "labels": val_labels},
        }


"""Helper functions.
"""


def extract_ids_from_dir(file_dir: str) -> list[str]:
    """Extract data ids from dataset directory."""
    ids = []
    for file in os.listdir(file_dir):
        if os.path.isfile(os.path.join(file_dir, file)):
            id, _ = os.path.splitext(file)
            ids.append(id)

    return ids
