from typing import Literal
import os
import yaml
import struct

import torch
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import Compose

from machine_learning.utils import print_dict


class FullLoadDataset(Dataset):
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


class LazyLoadDataset(Dataset):
    r"""延迟加载数据集.

    使用于大型数据集，减小内存空间占用，数据读取速度较慢.
    """

    def __init__(self):
        super().__init__()

    def __len__(self):
        pass

    def __getitem__(self, index):
        return super().__getitem__(index)


def minist_parser(dataset_dir: str, labels: bool = True) -> dict[str, np.ndarray]:
    """minist手写数字集数据解析函数

    Args:
        file_path (str): minist手写数字集储存路径.
        labels (bool): 是否加载标签数据.

    Returns:
        dict (str, np.ndarray): 训练集数据、训练集标签(可选)、验证集、验证集标签(可选).
    """
    returns = {}

    def load_idx3_ubyte(dataset_dir):
        with open(dataset_dir, "rb") as f:
            # 读取前16个字节的文件头信息
            magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))

            # 读取图像数据，并重新整形为(num_images, rows, cols)的三维数组
            images = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows, cols)

            return images, magic, num_images, rows, cols

    def load_idx1_ubyte(dataset_dir):
        with open(dataset_dir, "rb") as f:
            # 读取前8个字节的文件头信息
            magic, num_labels = struct.unpack(">II", f.read(8))

            # 读取标签数据，并重新整形为(num_labels,)的一维数组
            labels = np.fromfile(f, dtype=np.uint8)

            return labels, magic, num_labels

    dataset_dir = os.path.abspath(dataset_dir)
    train_data_dir = os.path.join(dataset_dir, "train")
    val_data_dir = os.path.join(dataset_dir, "test")
    print("[INFO] Train data directory path: ", train_data_dir)
    print("[INFO] Val data directory path: ", val_data_dir)

    # 加载数据
    train_data = load_idx3_ubyte(os.path.join(train_data_dir, "images_train.idx3-ubyte"))[0]
    val_data = load_idx3_ubyte(os.path.join(val_data_dir, "images_test.idx3-ubyte"))[0]

    returns["train_data"] = train_data
    returns["val_data"] = val_data

    if labels:
        train_labels = load_idx1_ubyte(os.path.join(train_data_dir, "labels_train.idx1-ubyte"))[0]
        val_labels = load_idx1_ubyte(os.path.join(val_data_dir, "labels_test.idx1-ubyte"))[0]

        returns["train_labels"] = train_labels
        returns["val_labels"] = val_labels

    return returns


def yolo_parser(dataset_dir: str) -> dict[str, list | np.ndarray]:
    """yolo格式数据集加载函数,返回image路径列表,标注则一次性返回.

    Args:
        file_path (str): yolo类型数据集位置.

    Returns:
        dict (str, np.ndarray): 训练集数据、训练集标签、验证集、验证集标签.
    """
    dataset_dir = os.path.abspath(dataset_dir)
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")

    metadata = yaml.safe_load(os.path.join(dataset_dir, "metadata.yaml"))
    dataset_name = metadata["dataset_name"]
    names_file = os.path.join(dataset_dir, metadata["names_file"])

    print(f"[INFO] Information of dataset {dataset_name}:")
    print_dict(metadata)


def coco_parser(file_path: str, purpose: Literal["captions", "instances", "keypoints"]) -> dict[str, np.ndarray]:
    pass


def voc_parser(
    file_path: str, purpose: Literal["detections", "segmentation_classes", "segmentation_objects"]
) -> dict[str, np.ndarray]:
    pass
