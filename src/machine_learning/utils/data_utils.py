from __future__ import annotations

import os
import struct
from dataclasses import dataclass, MISSING
from abc import ABC, abstractmethod
from typing import Literal, Any, Callable, Type

import torch
import numpy as np
from torchvision.transforms import Compose
from torch.utils.data import Dataset

from machine_learning.utils.others import print_dict, load_config_from_path, print_segmentation, list_from_txt


class FullDataset(Dataset):
    r"""完全加载数据集.

    使用于小型数据集，占用内存空间小，加快数据读取速度.
    """

    def __init__(
        self, data: np.ndarray | torch.Tensor, labels: np.ndarray | torch.Tensor = None, tansform: Compose = None
    ) -> None:
        """完全加载数据集初始化.

        Args:
            data (np.ndarray | torch.Tensor): 数据
            labels (np.ndarray | torch.Tensor, optional): 标签. Defaults to None.
            tansform (Compose, optional): 数据转换器. Defaults to None.
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
    r"""延迟加载数据集.

    使用于大型数据集，减小内存空间占用，数据读取速度较慢.
    """

    def __init__(
        self,
        dataset_type: str,
        data_info: dict[str, Any],
        img_size: 416,
        multiscale: bool = False,
        tansform: Compose = None,
    ):
        """LazyLoadDataset初始化.

        Args:
            dataset_type (str): 使用数据集的标注类型，比如"coco","yolo","voc"等.
            data_info (dict[str, Any]): 数据集的元信息.
            img_size (416): 图片形状大小.
            multiscale (bool, optional): 启用多尺度训练. Defaults to False.
            tansform (Compose, optional): 数据转换器. Defaults to None.
        """
        super().__init__()

    def __len__(self):
        pass

    def __getitem__(self, index):
        return super().__getitem__(index)


class ParserFactory:
    r"""工厂类, 用于生成具体的数据解析器, 遵循开闭原则."""

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
            print(f"DataLoaderFactory has registred dataset_parser {parser_cls.__name__}.")
            return parser_cls

        return parser_wrapper

    def parser_create(self, parser_cfg: ParserCfg) -> DatasetParser:
        dataset_dir = os.path.abspath(parser_cfg.dataset_dir)
        metadata = self._load_metadata(dataset_dir)

        dataset_type: str = metadata["dataset_type"]
        print(f"[INFO] Dataset type: {dataset_type}")
        # 动态获取解析器
        if dataset_type not in self._parser_registry:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
        parser_cls = self._parser_registry[dataset_type]

        return parser_cls(parser_cfg)

    def _load_metadata(self, dataset_dir: str) -> dict:
        """加载元数据文件"""
        metadata_path = os.path.join(dataset_dir, "metadata.yaml")
        return load_config_from_path(metadata_path)

    def __str__(self):
        return f"DataLoaderFactory(parsers={self.parsers})"


@dataclass
class ParserCfg:
    dataset_dir: str = MISSING
    labels: bool = MISSING
    data_load_method: Literal["full", "lazy"] = "full"


class DatasetParser(ABC):
    """数据集解析器抽象基类."""

    def __init__(self, parser_cfg: ParserCfg) -> None:
        super().__init__()
        self.cfg = parser_cfg
        self.data_load_method = parser_cfg.data_load_method
        self.labels = parser_cfg.labels
        self.dataset_dir = parser_cfg.dataset_dir

        self.train_data = None
        self.val_data = None
        self.train_labels = None
        self.val_labels = None

    @abstractmethod
    def parser(self) -> dict[str, Dataset]:
        pass

    def __str__(self) -> str:
        pass


@ParserFactory.register_parser("minist")
class MinistParser(DatasetParser):
    r"""minist手写数字集数据解析器, 由于misit数据集体量小, 只实现了完全解析."""

    def __init__(self, parser_cfg: ParserCfg):
        super().__init__(parser_cfg)

    @staticmethod
    def load_idx3_ubyte(dataset_dir: str) -> tuple:
        with open(dataset_dir, "rb") as f:
            # 读取前16个字节的文件头信息
            magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
            # 读取图像数据，并重新整形为(num_images, rows, cols)的三维数组
            images = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows, cols)

            return images, magic, num_images, rows, cols

    @staticmethod
    def load_idx1_ubyte(dataset_dir: str) -> tuple:
        with open(dataset_dir, "rb") as f:
            # 读取前8个字节的文件头信息
            magic, num_labels = struct.unpack(">II", f.read(8))
            # 读取标签数据，并重新整形为(num_labels,)的一维数组
            labels = np.fromfile(f, dtype=np.uint8)

            return labels, magic, num_labels

    def create(self, dataset_dir: str, transforms: Compose = None) -> dict[str, Dataset]:
        dataset_dir = os.path.abspath(self.dataset_dir)
        train_data_dir = os.path.join(dataset_dir, "train")
        val_data_dir = os.path.join(dataset_dir, "test")
        print("[INFO] Train data directory path: ", train_data_dir)
        print("[INFO] Val data directory path: ", val_data_dir)

        # 加载数据
        train_data = self.load_idx3_ubyte(os.path.join(train_data_dir, "images_train.idx3-ubyte"))[0]
        val_data = self.load_idx3_ubyte(os.path.join(val_data_dir, "images_test.idx3-ubyte"))[0]

        self.train_data = train_data
        self.val_data = val_data

        if self.cfg.labels:
            train_labels = self.load_idx1_ubyte(os.path.join(train_data_dir, "labels_train.idx1-ubyte"))[0]
            val_labels = self.load_idx1_ubyte(os.path.join(val_data_dir, "labels_test.idx1-ubyte"))[0]

            self.train_labels = train_labels
            self.val_labels = val_labels

        self.trian_dataset = FullDataset(self.train_data, self.train_labels, transforms)
        self.val_dataset = FullDataset(self.val_data, self.val_labels, transforms)

        return {"train": self.trian_dataset, "val": self.val_dataset}


@ParserFactory.register_parser("yolo")
class YoloParser(DatasetParser):
    r"""yolo格式数字集解析器"""

    def __init__(self, dataset_dir: str, labels: bool = True):
        super().__init__(dataset_dir)
        self.labels = labels

    @staticmethod
    def load_idx3_ubyte(dataset_dir: str) -> tuple:
        with open(dataset_dir, "rb") as f:
            # 读取前16个字节的文件头信息
            magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
            # 读取图像数据，并重新整形为(num_images, rows, cols)的三维数组
            images = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows, cols)

            return images, magic, num_images, rows, cols

    @staticmethod
    def load_idx1_ubyte(dataset_dir: str) -> tuple:
        with open(dataset_dir, "rb") as f:
            # 读取前8个字节的文件头信息
            magic, num_labels = struct.unpack(">II", f.read(8))
            # 读取标签数据，并重新整形为(num_labels,)的一维数组
            labels = np.fromfile(f, dtype=np.uint8)

            return labels, magic, num_labels

    def create(self, transforms: Compose = None) -> dict[str, Dataset]:
        dataset_dir = os.path.abspath(self.dataset_dir)
        train_data_dir = os.path.join(dataset_dir, "train")
        val_data_dir = os.path.join(dataset_dir, "test")
        print("[INFO] Train data directory path: ", train_data_dir)
        print("[INFO] Val data directory path: ", val_data_dir)

        # 加载数据
        train_data = self.load_idx3_ubyte(os.path.join(train_data_dir, "images_train.idx3-ubyte"))[0]
        val_data = self.load_idx3_ubyte(os.path.join(val_data_dir, "images_test.idx3-ubyte"))[0]

        self.train_data = train_data
        self.val_data = val_data

        if self.labels:
            train_labels = self.load_idx1_ubyte(os.path.join(train_data_dir, "labels_train.idx1-ubyte"))[0]
            val_labels = self.load_idx1_ubyte(os.path.join(val_data_dir, "labels_test.idx1-ubyte"))[0]

            self.train_labels = train_labels
            self.val_labels = val_labels

        self.trian_dataset = FullDataset(self.train_data, self.train_labels, transforms)
        self.val_dataset = FullDataset(self.val_data, self.val_labels, transforms)

        return {"train": self.trian_dataset, "val": self.val_dataset}


def yolo_parser(dataset_dir: str) -> dict[str, list]:
    """yolo格式数据集加载函数,返回image和labels列表.

    Args:
        file_path (str): yolo类型数据集位置.

    Returns:
        dict (str, list | np.ndarray): 返回数据.
    """

    dataset_dir = os.path.abspath(dataset_dir)

    metadata = load_config_from_path(os.path.join(dataset_dir, "metadata.yaml"))

    dataset_name = metadata["dataset_name"]
    if metadata["dataset_type"] != "yolo":
        raise TypeError(f"Dataset {dataset_name} is not the type of yolo.")

    class_names_file = os.path.join(dataset_dir, metadata["names_file"])

    print_segmentation()
    print("Information of dataset:")
    print_dict(metadata)
    print_segmentation()

    # 读取种类名称
    class_names = list_from_txt(class_names_file)

    train_img_dir = os.path.join(dataset_dir, "images/trian")
    val_img_dir = os.path.join(dataset_dir, "images/val")
    train_labels_dir = os.path.join(dataset_dir, "labels/train")
    val_labels_dir = os.path.join(dataset_dir, "labels/val")

    # 读取训练、验证图像列表
    train_img_ls = list_from_txt(dataset_dir + "/images_train.txt")
    val_img_ls = list_from_txt(dataset_dir + "/images_val.txt")
    train_labels_ls = [img.rsplit(".", 1)[0] + ".txt" for img in train_img_ls]
    val_labels_ls = [img.rsplit(".", 1)[0] + ".txt" for img in val_img_ls]

    # 添加绝对路径
    train_img_path_ls = [train_img_dir + img for img in train_img_ls]
    val_img_path_ls = [val_img_dir + img for img in val_img_ls]
    train_labels_path_ls = [train_labels_dir + label for label in train_labels_ls]
    val_labels_path_ls = [val_labels_dir + label for label in val_labels_ls]

    return {
        "metadata": metadata,
        "class_names": class_names,
        "train_images_path_list": train_img_path_ls,
        "val_images_path_list": val_img_path_ls,
        "train_labels_path_list": train_labels_path_ls,
        "val_labels_path_list": val_labels_path_ls,
    }


def coco_parser(file_path: str, purpose: Literal["captions", "instances", "keypoints"]) -> dict[str, np.ndarray]:
    pass


def voc_parser(
    file_path: str, purpose: Literal["detections", "segmentation_classes", "segmentation_objects"]
) -> dict[str, np.ndarray]:
    pass
