from __future__ import annotations

import os
import struct
import warnings
from PIL import Image, ImageFile
from dataclasses import dataclass, MISSING
from abc import ABC, abstractmethod
from typing import Literal, Callable, Sequence, Any

import torch
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import Compose
from torch.utils.data import Dataset

from machine_learning.utils.others import print_dict, load_config_from_path, print_segmentation, list_from_txt


class FullDataset(Dataset):
    r"""完全加载数据集.

    使用于小型数据集，占用内存空间小，加快数据读取速度.
    """

    def __init__(
        self,
        data: np.ndarray | torch.Tensor,
        labels: np.ndarray | torch.Tensor = None,
        tansform: Compose = None,
    ) -> None:
        """完全加载数据集初始化.

        Args:
            data (np.ndarray | torch.Tensor): 数据
            labels (np.ndarray | torch.Tensor, optional): 标签. Defaults to None.
            tansforms (Compose, optional): 数据转换器. Defaults to None.
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
        img_paths: Sequence[str],
        label_paths: Sequence[int],
        img_size: int = 416,
        multiscale: bool = False,
        transform: Compose = None,
    ):
        """LazyLoadDataset初始化.

        Args:
            img_paths (Sequence[str]): 图片地址列表.
            label_paths: (Sequence[int]): 标签地址列表.
            img_size (416): 图片形状大小.
            multiscale (bool, optional): 启用多尺度训练. Defaults to False.
            transform (Compose, optional): 数据转换器. Defaults to None.
        """
        super().__init__()

        self.img_paths = img_paths
        self.label_paths = label_paths
        self.img_size = img_size
        self.multiscale = multiscale
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        #  Image
        try:
            img_path = self.img_paths[index % len(self.img_paths)]
            img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)

        except Exception:
            print(f"Could not read image '{img_path}'.")
            return

        #  Label
        try:
            # 有些image没有对应的label
            label_path = self.label_paths[index % len(self.img_paths)]

            # Ignore warning if file is empty
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                boxes = np.loadtxt(label_path).reshape(-1, 5)

        except Exception:
            print(f"Could not read label '{label_path}'.")
            return

        #  Transform
        if self.transform:
            try:
                img, bb_targets = self.transform((img, boxes))
            except Exception:
                print("Could not apply transform.")
                return

        return img_path, img, bb_targets


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
            print(f"DataLoaderFactory has registred dataset_parser '{parser_cls.__name__}'.")
            return parser_cls

        return parser_wrapper

    def parser_create(self, parser_cfg: ParserCfg) -> DatasetParser:
        dataset_dir = os.path.abspath(parser_cfg.dataset_dir)
        metadata = self._load_metadata(dataset_dir)

        dataset_type: str = metadata["dataset_type"]
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
    transforms: Compose | None = None


class DatasetParser(ABC):
    """数据集解析器抽象基类."""

    def __init__(self, parser_cfg: ParserCfg) -> None:
        super().__init__()
        self.cfg = parser_cfg

        self.dataset_dir = parser_cfg.dataset_dir
        self.labels = parser_cfg.labels
        self.data_load_method = parser_cfg.data_load_method
        self.transforms = self.cfg.transforms

    @abstractmethod
    def parse(self) -> Any:
        pass

    @abstractmethod
    def create(self) -> dict[str, Dataset]:
        """根据数据集的解析数据信息创建数据集.

        Returns:
            dict[str, Dataset]: 返回包含训练(train)和验证(val)数据集的字典.
        """
        pass

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"dataset_dir={self.dataset_dir}, "
            f"labels={self.labels}, "
            f"data_load_method={self.data_load_method}, "
            f"transforms={self.transforms})"
        )


@ParserFactory.register_parser("minist")
class MinistParser(DatasetParser):
    r"""minist手写数字集数据解析器, 由于misit数据集体量小, 只实现了完全解析."""

    def __init__(self, parser_cfg: ParserCfg):
        super().__init__(parser_cfg)

    def parse(self) -> None:
        metadata = load_config_from_path(os.path.join(self.dataset_dir, "metadata.yaml"))

        dataset_name = metadata["dataset_name"]
        if metadata["dataset_type"] != "minist":
            raise TypeError(f"Dataset {dataset_name} is not the type of minist.")

        print_segmentation()
        print("Information of dataset:")
        print_dict(metadata)
        print_segmentation()

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

    def create(self) -> dict[str, Dataset]:
        """根据MinistParser的配置信息创建数据集.

        Returns:
            dict[str, Dataset]: 返回包含训练(train)和验证(val)数据集的字典.
        """
        self.parse()

        train_data_dir = os.path.join(self.dataset_dir, "train")
        val_data_dir = os.path.join(self.dataset_dir, "test")
        print("[INFO] Train data directory path: ", train_data_dir)
        print("[INFO] Val data directory path: ", val_data_dir)

        # 加载数据
        train_data = self.load_idx3_ubyte(os.path.join(train_data_dir, "images_train.idx3-ubyte"))[0]
        val_data = self.load_idx3_ubyte(os.path.join(val_data_dir, "images_test.idx3-ubyte"))[0]

        if self.cfg.labels:
            train_labels = self.load_idx1_ubyte(os.path.join(train_data_dir, "labels_train.idx1-ubyte"))[0]
            val_labels = self.load_idx1_ubyte(os.path.join(val_data_dir, "labels_test.idx1-ubyte"))[0]
        else:
            train_labels, val_labels = None, None

        trian_dataset = FullDataset(train_data, train_labels, self.transforms)
        val_dataset = FullDataset(val_data, val_labels, self.transforms)

        return {"train": trian_dataset, "val": val_dataset}


@ParserFactory.register_parser("yolo")
class YoloParser(DatasetParser):
    r"""yolo格式数字集解析器"""

    def __init__(self, parser_cfg: ParserCfg):
        super().__init__(parser_cfg)

    def parse(self) -> tuple:
        metadata = load_config_from_path(os.path.join(self.dataset_dir, "metadata.yaml"))

        dataset_name = metadata["dataset_name"]
        if metadata["dataset_type"] != "yolo":
            raise TypeError(f"Dataset {dataset_name} is not the type of yolo.")

        class_names_file = os.path.join(self.dataset_dir, metadata["names_file"])

        print_segmentation()
        print("Information of dataset:")
        print_dict(metadata)
        print_segmentation()

        # 读取种类名称
        classes = list_from_txt(class_names_file)

        train_img_dir = os.path.join(self.dataset_dir, "images/trian")
        val_img_dir = os.path.join(self.dataset_dir, "images/val")
        train_labels_dir = os.path.join(self.dataset_dir, "labels/train")
        val_labels_dir = os.path.join(self.dataset_dir, "labels/val")

        # 读取训练、验证图像列表
        train_img_ls = list_from_txt(self.dataset_dir + "/images_train.txt")
        val_img_ls = list_from_txt(self.dataset_dir + "/images_val.txt")
        train_labels_ls = [img.rsplit(".", 1)[0] + ".txt" for img in train_img_ls]
        val_labels_ls = [img.rsplit(".", 1)[0] + ".txt" for img in val_img_ls]
        print(len(train_img_ls), len(train_labels_ls))
        print(len(val_img_ls), len(val_labels_ls))

        # 添加绝对路径
        train_img_paths = [train_img_dir + img for img in train_img_ls]
        val_img_paths = [val_img_dir + img for img in val_img_ls]
        train_labels_paths = [train_labels_dir + label for label in train_labels_ls]
        val_labels_paths = [val_labels_dir + label for label in val_labels_ls]

        return classes, train_img_paths, val_img_paths, train_labels_paths, val_labels_paths

    def create(self, transforms: Compose = None) -> dict[str, Dataset]:
        """根据YoloParser的配置信息创建数据集.

        Returns:
            dict[str, Dataset]: 返回包含训练(train)和验证(val)数据集的字典.
        """
        # 解析类别和路径信息
        classes, train_img_paths, val_img_paths, train_labels_paths, val_labels_paths = self.parse()


@ParserFactory.register_parser("coco")
class CocoParser(DatasetParser):
    r"""coco格式数字集解析器"""

    def __init__(self, parser_cfg: ParserCfg):
        super().__init__(parser_cfg)

    @staticmethod
    def parse(dataset_dir) -> tuple:
        pass

    def create(self, transforms: Compose = None) -> dict[str, Dataset]:
        pass


@ParserFactory.register_parser("voc")
class VocParser(DatasetParser):
    r"""voc格式数字集解析器"""

    def __init__(self, parser_cfg: ParserCfg):
        super().__init__(parser_cfg)

    @staticmethod
    def parse(dataset_dir) -> tuple:
        pass

    def create(self, transforms: Compose = None) -> dict[str, Dataset]:
        pass


"""
Helper function
"""


def pad_to_square(img: torch.Tensor, pad_value: float | None = None) -> tuple:
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # 填充数值
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # 填充数 (左， 右， 上， 下， 前， 后)
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)

    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image: torch.Tensor, size) -> torch.Tensor:
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def rescale_boxes(boxes: torch.Tensor, current_dim: int, original_shape: tuple[int]) -> torch.Tensor:
    """
    将目标检测模型输出的边界框坐标从调整后的正方形图像尺寸转换回原始图像尺寸,
    [example](/home/yangxf/WorkSpace/machine_learning/docs/pictures/01.jpg)
    """
    orig_h, orig_w = original_shape

    # 计算增加的pad, 应对pad后放缩的情况
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))

    # 移除pad后的尺寸
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x

    # 重新映射边界框
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h

    return boxes


def xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2

    return y


def xywh2xyxy_np(x):
    y = np.zeros_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def xyxy2xywh(x: torch.Tensor) -> torch.Tensor:
    y = x.new(x.shape)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]

    return y
