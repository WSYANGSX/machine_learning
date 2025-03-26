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


def data_parse(file_path: str) -> tuple:
    file_path = os.path.abspath(file_path)
    train_data_path = os.path.join(file_path, "train")
    validate_data_path = os.path.join(file_path, "test")
    print("[INFO] Train data path: ", train_data_path)
    print("[INFO] Validate data path: ", validate_data_path)

    # 加载数据
    train_data, train_labels = (
        load_idx3_ubyte(os.path.join(train_data_path, "images_train.idx3-ubyte"))[0],
        load_idx1_ubyte(os.path.join(train_data_path, "labels_train.idx1-ubyte"))[0],
    )
    validate_data, validate_labels = (
        load_idx3_ubyte(os.path.join(validate_data_path, "images_test.idx3-ubyte"))[0],
        load_idx1_ubyte(os.path.join(validate_data_path, "labels_test.idx1-ubyte"))[0],
    )
    print(
        "[INFO] train data shape: ",
        train_data.shape,
        " " * 5,
        "train labels shape: ",
        train_labels.shape,
        " " * 5,
        "test data shape: ",
        validate_data.shape,
        " " * 5,
        "test labels shape: ",
        validate_labels.shape,
    )

    return train_data, train_labels, validate_data, validate_labels
