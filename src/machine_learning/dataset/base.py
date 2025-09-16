from typing import Any, Union, Literal

import os
import cv2
import math
import random
import psutil
import warnings
import numpy as np

from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from pympler import asizeof
from torch.utils.data import Dataset
from multiprocessing.pool import ThreadPool
from torchvision.transforms import Compose, ToTensor, Normalize

from machine_learning.utils.logger import LOGGER
from machine_learning.types.aliases import FilePath
from machine_learning.utils.constants import NUM_THREADS, IMG_TYPES


class DatasetBase(Dataset):
    """
    Base dataset class for loading and processing data.

    Args:
        data (list[str] | np.ndarray): Path list to the data or data itself with np.ndarray format.
        labels (list[str] | np.ndarray): Path list to the labels or labels itself with np.ndarray format.
        cache (bool, optional): Cache data to RAM or disk during training. Defaults to False.
        augment (bool, optional): If True, data augmentation is applied. Defaults to True.
        hyp (dict, optional): Hyperparameters to apply data augmentation. Defaults to None.
        batch_size (int, optional): Size of batches. Defaults to None.
        fraction (float): Fraction of dataset to utilize. Default is 1.0 (use all data).
    """

    def __init__(
        self,
        data: np.ndarray | list[str],
        labels: np.ndarray | list[str],
        cache: bool = False,
        augment: bool = True,
        hyp: dict[str, Any] | None = None,
        batch_size: int = 16,
        fraction: float = 1.0,
        mode: Literal["train", "val", "test"] = "train",
    ):
        """Initialize DatasetBase with given configuration and options."""
        super().__init__()

        self.augment = augment
        self.fraction = fraction
        self.batch_size = batch_size
        self.cache = cache.lower() if isinstance(cache, str) else "ram" if cache is True else None
        self.length = math.ceil(len(labels) * self.fraction)
        self.mode = mode

        # init buffers
        # labels
        self.labels = [None] * self.length
        self.label_files = [None] * self.length
        # data
        self.data = [None] * self.length
        self.data_files = [None] * self.length
        self.data_npy_files = [None] * self.length

        # cache labels and data
        # labels
        if isinstance(labels, np.ndarray):
            self.get_labels_np(labels)
        elif isinstance(labels, list):
            self.label_files = labels[: self.length]
            self.get_labels()
        else:
            raise TypeError(f"The input labels must be of np.ndarray or list type, but got {type(data)}.")

        # data
        if isinstance(data, np.ndarray):
            self.cache_data_np(data)
        elif isinstance(data, list):
            self.data_files = data[: self.length]
            self.data_npy_files = [Path(f).with_suffix(".npy") for f in self.data_files]

            # cache data
            if self.cache == "ram" and self.check_cache_ram():
                self.cache_data()
            elif self.cache == "disk" and self.check_cache_disk():
                self.cache_data()
        else:
            raise TypeError(f"The input data must be of np.ndarray or list type, but got {type(data)}.")

        # Buffer thread for data fusion
        self.buffer = []
        self.max_buffer_length = min((self.length, self.batch_size * 8, 1000)) if self.augment else 0

        # Transforms
        self.transforms = self.build_transforms(hyp=hyp)

    def get_labels(self) -> None:
        """Get labels from path list to buffers."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        LOGGER.info(f"Caching {self.mode} labels...")
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(self.cache_labels, range(self.length))
            pbar = tqdm(enumerate(results), total=self.length)
            for _, size in pbar:
                b += size
                pbar.desc = f"Caching {self.mode} labels ({b / gb:.5f}GB)"
            pbar.close()

    def get_labels_np(self, labels: np.ndarray) -> None:
        """Get labels from matrix input to buffers."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        LOGGER.info(f"Caching {self.mode} labels from matrix input...")
        pbar = tqdm(enumerate(labels), total=self.length)
        for i, label in pbar:
            label = self.label_format(label)
            self.labels[i] = label
            b += asizeof.asizeof(self.labels[i])
            pbar.desc = f"Caching {self.mode} labels ({b / gb:.5f}GB)"
        pbar.close()

    def cache_labels(self, i: int) -> float:
        """Cache label from paths to ram for faster loading."""
        f = self.label_files[i]
        label = self.lread(f)
        label = self.label_format(label)
        self.labels[i] = label

        return asizeof.asizeof(label)

    def cache_data_np(self, data: np.ndarray) -> None:
        """Cache np.ndarray format data to buffers"""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        LOGGER.info(f"Caching {self.mode} data from matrix input...")
        pbar = tqdm(enumerate(data), total=self.length)
        for i, data in pbar:
            self.data[i] = data
            b += self.data[i].nbytes
            pbar.desc = f"Caching {self.mode} data ({b / gb:.5f}GB)"
        pbar.close()

    def cache_data(self) -> None:
        """Cache data to memory or disk."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        fcn, storage = (self.cache_data_to_ram, "RAM") if self.cache == "ram" else (self.cache_data_to_disk, "Disk")
        LOGGER.info(f"Caching {self.mode} data to {storage}...")
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(self.length))
            pbar = tqdm(enumerate(results), total=self.length)
            for _, size in pbar:
                b += size
                pbar.desc = f"Caching {self.mode} data ({b / gb:.5f}GB {storage})"
            pbar.close()

    def cache_data_to_ram(self, i: int) -> float:
        """Cache data from paths or .npy file to ram for faster loading."""
        f, fn = self.data_files[i], self.data_npy_files[i]

        if fn.exists():
            try:
                data = np.load(fn)
            except Exception as e:
                LOGGER.warning(f"Removing corrupt *.npy image file {fn} due to: {e}")
                Path(fn).unlink(missing_ok=True)
                data = self.file_read(f)
        else:
            data = self.file_read(f)

        if data is not None:
            self.data[i] = data
            return self.data[i].nbytes
        else:
            return 0.0

    def cache_data_to_disk(self, i: int) -> float:
        """Cache data from paths to disk with as an .npy file for faster loading."""
        f = self.data_npy_files[i]
        if not f.exists():
            data = self.file_read(self.data_files[i])
            if data is not None:
                np.save(f, data, allow_pickle=False)
                return self.data_npy_files[i].stat().st_size
            else:
                return 0.0

    def load_data(self, i: int) -> tuple[Union[np.ndarray, None], Union[np.ndarray, None]]:
        """Loads 1 data and label from dataset index 'i', returns (data, label)."""
        dt, dtf, dtfn = self.data[i], self.data_files[i], self.data_npy_files[i]

        # load data
        if dt is not None:  # cached in RAM
            data = dt
        elif dtfn.exists():  # cached in Disk
            try:
                data = np.load(dtfn)
            except Exception as e:
                LOGGER.warning(f"Removing corrupt *.npy image file {dtfn} due to: {e}")
                Path(dtfn).unlink(missing_ok=True)
                data = self.file_read(dtf)
        else:
            data = self.file_read(dtf)

        # Add to buffer if training with augmentations
        if self.augment:
            self.data[i] = data
            self.buffer.append(i)
            if len(self.buffer) >= self.max_buffer_length:  # prevent empty buffer
                j = self.buffer.pop(0)
                if self.cache != "ram":
                    self.data[j] = None

        return data

    def __len__(self):
        """Returns the length of the labels list for the dataset."""
        return self.length

    def __getitem__(self, index: int) -> tuple[Union[np.ndarray, None], Union[np.ndarray, None]]:
        """Returns transformed label information for given index."""
        data, label = self.get_data_and_label(index)
        if self.transforms is not None:
            data = self.transforms(data)
        return data, label

    def get_data_and_label(self, index: int) -> tuple[Union[np.ndarray, None], Union[np.ndarray, None]]:
        label = deepcopy(self.labels[index])
        data = self.load_data(index)
        label = self.update_labels_info(label)

        return data, label

    def check_cache_ram(self, safety_margin: float = 0.5) -> bool:
        """Check data caching requirements vs available memory."""
        b, gb = 0, 1 << 30  # bytes of cached data, bytes per gigabytes
        n = min(self.length, 100)  # extrapolate from 100 random data
        skips = 0
        for _ in range(n):
            i = random.randint(0, self.length - 1)
            data = self.file_read(self.data_files[i])
            if data is None:
                skips += 1
                continue
            b += data.nbytes
        mem_required = b * self.length / (n - skips) * (1 + safety_margin)  # GB required to cache data into RAM
        mem = psutil.virtual_memory()
        if mem_required > mem.available:
            self.cache = None
            LOGGER.info(
                f"{mem_required / gb:.1f}GB RAM required to cache data "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, not caching data."
            )
            return False
        return True

    def check_cache_disk(self, safety_margin: float = 0.5) -> bool:
        """Check data caching requirements vs available disk space."""
        import shutil

        b, gb = 0, 1 << 30  # bytes of cached data, bytes per gigabytes
        n = min(self.length, 100)  # extrapolate from 30 random data
        skips = 0
        for _ in range(n):
            i = random.randint(0, self.length - 1)
            data = self.file_read(self.data_files[i])
            if data is None:
                skips += 1
                continue
            b += data.nbytes
            if not os.access(Path(self.data_files[i]).parent, os.W_OK):
                self.cache = None
                LOGGER.info("Skipping caching data to disk, directory not writeable.")
                return False
        disk_required = b * self.length / (n - skips) * (1 + safety_margin)  # bytes required to cache data to disk
        total, _, free = shutil.disk_usage(Path(self.data_files[0]).parent.parent)
        if disk_required > free:
            self.cache = None
            LOGGER.info(
                f"{disk_required / gb:.1f}GB disk space required, "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{free / gb:.1f}/{total / gb:.1f}GB free, not caching data to disk."
            )
            return False
        return True

    def lread(self, path: FilePath) -> dict[str, Any]:
        """Users implement their own label reading logic, and label format.

        Args:
            path (FilePath): label file path.

        Returns:
            dict[str, Any]: label.
        """
        label = self.file_read(path)

        return label

    def update_labels_info(self, label) -> dict[str, Any]:
        """Custom your label format here."""
        return label

    @staticmethod
    def file_read(path: FilePath) -> np.ndarray | None:
        path = Path(path)
        extension = path.suffix.lower()

        if not path.exists():
            LOGGER.error(f"File does not exist: {path}")
            return None

        try:
            if extension in IMG_TYPES:  # imgs
                data = cv2.imread(str(path), cv2.IMREAD_COLOR_RGB)  # rgb
                if data is None:
                    LOGGER.error("Image Not Found.")
                    return None
                return data

            elif extension == ".txt":  # text
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    data = np.loadtxt(path, dtype=np.float32)
                return data

            elif extension == ".npy":  # numpy
                data = np.load(path)
                return data

            else:
                LOGGER.error(
                    f"Unsupported file type: {extension}. Supported types: {list(IMG_TYPES)} + ['.txt', '.npy']"
                )
                return None

        except Exception as e:
            LOGGER.error(f"Could not read file '{path}': {e}")
            return None

    def label_format(self, label: np.ndarray) -> dict[str, Any]:
        """format the label from np.ndarray to a custom form."""
        label = {"label": label}  # Customize
        return label

    def build_transforms(self, hyp=None):
        """
        Users can customize augmentations here.

        Example:
            ```python
            if self.augment:
                # Training transforms
                return Compose([])
            else:
                # Val transforms
                return Compose([])
            ```
        """
        return Compose([ToTensor(), Normalize(hyp.get("mean", 0.0), hyp.get("std", 1.0))])
