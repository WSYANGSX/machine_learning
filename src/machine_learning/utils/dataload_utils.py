from __future__ import annotations

import os
import struct
import random
import warnings
from copy import deepcopy
from PIL import Image, ImageFile
from abc import ABC, abstractmethod
from dataclasses import dataclass, MISSING
from typing import Callable, Sequence, Any

import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset

from machine_learning.utils.transform import CustomTransform, YoloTransform
from machine_learning.utils.image import resize
from machine_learning.utils.others import print_dict, load_config_from_path, print_segmentation, list_from_txt


ImageFile.LOAD_TRUNCATED_IMAGES = True


class FullDataset(Dataset):
    r"""完全加载数据集.

    使用于小型数据集，占用内存空间小，加快数据读取速度.
    """

    def __init__(
        self,
        data: np.ndarray | torch.Tensor,
        labels: np.ndarray | torch.Tensor = None,
        tansform: transforms.Compose | CustomTransform = None,
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
        data_paths: Sequence[str],
        label_paths: Sequence[int],
        transform: transforms.Compose | CustomTransform = None,
    ):
        """LazyLoadDataset初始化.

        Args:
            data_paths (Sequence[str]): 数据地址列表.
            label_paths: (Sequence[int]): 标签地址列表.
            transform (Compose, optional): 数据增强转换器. Defaults to None.
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
    r"""yolo目标检测类数据集.

    目标检测数据集一般较大，继承延迟加载数据集，减小内存空间占用，数据读取速度较慢.
    """

    def __init__(
        self,
        img_paths: Sequence[str],
        label_paths: Sequence[int],
        transform: YoloTransform = None,
        img_size: int = 416,
        multiscale: bool = False,
    ):
        """LazyLoadDataset初始化.

        Args:
            data_paths (Sequence[str]): 数据地址列表.
            label_paths: (Sequence[int]): 标签地址列表.
            transform (Compose, optional): 数据增强转换器. Defaults to None.
        """
        super().__init__(data_paths=img_paths, label_paths=label_paths, transform=transform)

        self.img_size = img_size
        self.multiscale = multiscale

        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
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
                labels = np.loadtxt(label_path).reshape(-1, 5)
                bboxes = labels[:, 1:5]
                category_ids = np.array(labels[:, 0], dtype=np.uint16)

        except Exception:
            print(f"Could not read label '{label_path}'.")
            return

        #  Transform
        if self.transform:
            try:
                img, bboxes, category_ids = self.transform((img, bboxes, category_ids))
            except Exception:
                print("Could not apply transform.")
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
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))

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
    transforms: transforms.Compose | CustomTransform | None = None


class DatasetParser(ABC):
    """数据集解析器抽象基类."""

    def __init__(self, parser_cfg: ParserCfg) -> None:
        super().__init__()
        self.cfg = parser_cfg

        self.dataset_dir = parser_cfg.dataset_dir
        self.labels = parser_cfg.labels
        self.transforms = self.cfg.transforms

    @abstractmethod
    def parse(self) -> dict[str, Any]:
        """解析数据集元信息

        Returns:
            dict[str, Any]: 元信息返回值.
        """
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
            f"transforms={self.transforms})"
        )


@ParserFactory.register_parser("minist")
class MinistParser(DatasetParser):
    r"""minist手写数字集数据解析器, 由于misit数据集体量小, 只实现了完全解析."""

    def __init__(self, parser_cfg: ParserCfg):
        super().__init__(parser_cfg)

    def parse(self) -> dict[str, Any]:
        """ """
        metadata = load_config_from_path(os.path.join(self.dataset_dir, "metadata.yaml"))

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

    def parse(self) -> dict[str, Any]:
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

        train_img_dir = os.path.join(self.dataset_dir, "images/train/")
        val_img_dir = os.path.join(self.dataset_dir, "images/val/")
        train_labels_dir = os.path.join(self.dataset_dir, "labels/train/")
        val_labels_dir = os.path.join(self.dataset_dir, "labels/val/")

        # 读取训练、验证图像列表
        train_img_ls = list_from_txt(self.dataset_dir + "/images_train.txt")
        val_img_ls = list_from_txt(self.dataset_dir + "/images_val.txt")
        train_labels_ls = [img.rsplit(".", 1)[0] + ".txt" for img in train_img_ls]
        val_labels_ls = [img.rsplit(".", 1)[0] + ".txt" for img in val_img_ls]

        # 添加绝对路径
        train_img_paths = [train_img_dir + img for img in train_img_ls]
        val_img_paths = [val_img_dir + img for img in val_img_ls]
        train_labels_paths = [train_labels_dir + label for label in train_labels_ls]
        val_labels_paths = [val_labels_dir + label for label in val_labels_ls]

        return {
            "class_names": classes,
            "train_img_paths": train_img_paths,
            "val_img_paths": val_img_paths,
            "train_labels_paths": train_labels_paths,
            "val_labels_paths": val_labels_paths,
        }

    def create(self) -> dict[str, Dataset]:
        """根据YoloParser的配置信息创建数据集.

        Returns:
            dict[str, Dataset]: 返回包含训练(train)和验证(val)数据集的字典.
        """
        # 解析类别和路径信息
        metadata = self.parse()

        trian_dataset = YoloDataset(metadata["train_img_paths"], metadata["train_labels_paths"], self.transforms)
        val_dataset = YoloDataset(metadata["val_img_paths"], metadata["val_labels_paths"], self.transforms)

        return {"train": trian_dataset, "val": val_dataset}
