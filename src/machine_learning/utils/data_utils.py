from typing import Literal

import os
import struct
import numpy as np

from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data, labels=None, tansform=None) -> None:
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


def minist_parse(file_path: str, labels: bool = True) -> dict[str, np.ndarray]:
    """minist手写数字集数据解析函数

    Args:
        file_path (str): minist手写数字集储存路径.
        labels (bool): 是否加载标签数据.

    Returns:
        dict (str, np.ndarray): 训练集数据、训练集标签(可选)、验证集、验证集标签(可选).
    """

    def load_idx3_ubyte(file_path):
        with open(file_path, "rb") as f:
            # 读取前16个字节的文件头信息
            magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))

            # 读取图像数据，并重新整形为(num_images, rows, cols)的三维数组
            images = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows, cols)

            return images, magic, num_images, rows, cols

    def load_idx1_ubyte(file_path):
        with open(file_path, "rb") as f:
            # 读取前8个字节的文件头信息
            magic, num_labels = struct.unpack(">II", f.read(8))

            # 读取标签数据，并重新整形为(num_labels,)的一维数组
            labels = np.fromfile(f, dtype=np.uint8)

            return labels, magic, num_labels

    file_path = os.path.abspath(file_path)
    train_data_path = os.path.join(file_path, "train")
    validate_data_path = os.path.join(file_path, "test")
    print("[INFO] Train data path: ", train_data_path)
    print("[INFO] Validate data path: ", validate_data_path)

    # 加载数据
    train_data = load_idx3_ubyte(os.path.join(train_data_path, "images_train.idx3-ubyte"))[0]
    validate_data = load_idx3_ubyte(os.path.join(validate_data_path, "images_test.idx3-ubyte"))[0]

    if labels:
        train_labels = load_idx1_ubyte(os.path.join(train_data_path, "labels_train.idx1-ubyte"))[0]
        validate_labels = load_idx1_ubyte(os.path.join(validate_data_path, "labels_test.idx1-ubyte"))[0]

        return {
            "train_data": train_data,
            "train_labels": train_labels,
            "validate_data": validate_data,
            "validate_labels": validate_labels,
        }

    return {"train_data": train_data, "train_labels": train_labels}


def yolo_parse(file_path: str) -> dict[str, np.ndarray]:
    """yolo格式数据集加载函数

    Args:
        file_path (str): yolo类型数据集位置.

    Returns:
        dict (str, np.ndarray): 训练集数据、训练集标签、验证集、验证集标签.
    """
    pass


def coco_parse(file_path: str, purpose: Literal["captions", "instances", "keypoints"]) -> dict[str, np.ndarray]:
    pass


def voc_parse(
    file_path: str, purpose: Literal["detections", "segmentation_classes", "segmentation_objects"]
) -> dict[str, np.ndarray]:
    pass
