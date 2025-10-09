from typing import Any, Union, Literal

import os
import cv2
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
from machine_learning.utils.constants import NUM_THREADS, IMG_FORMATS


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
        mode (Literal["train", "val", "test"]): The mode of the dataset.

    Properties:
        data (list[np.ndarray]): List of data.
        data_files (list[str]): List of data file paths.
        data_npy_files (list[str]): List of data .npy file paths.
        labels (list[dict[str, Any]]): List of label data dictionaries.
        label_files (list[str]): List of label file paths.
        length (int): Number of data in the dataset.
        transforms (callable): Transformation function.

    Methods:
        get_labels(): Get the labels of the dataset.
        get_labels_np(np.ndarray): Get the labels of the dataset from np.ndarray label source.
        cache_labels(int): Cache labels from label file paths and format labels in a custom style using lread() and label_format().
        cache_data_np(np.ndarray): Cache the data of the dataset from np.ndarray data source.
        cache_data(int): Cache data from data file paths.
        cache_data_to_ram(int): Cache data to RAM.
        cache_data_to_disk(int): Cache data to Disk.
        load_data(int): Load a data.
        __len__(): Return the number of items in the dataset.
        __getitem__(int): Returns transformed label information for given index.
        get_data_and_label(int): Returns Data and label information for given index.
        update_labels_info(dict[str, Any]): Update labels in get_data_and_label().
        check_cache_ram(float): Check data caching requirements vs available memory.
        check_cache_disk(float): Check data caching requirements vs disk free space.
        lread(FilePath): Load labels from label file path by a certain logic.
        label_format(np.ndarray | None): Return the label in the custom format dictionaries.
        build_transforms(dict[str, Any]): Build data transform.

    """

    def __init__(
        self,
        data: np.ndarray | list[str],
        labels: np.ndarray | list[str],
        cache: bool = False,
        augment: bool = True,
        hyp: dict[str, Any] = {},
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
        self.mode = mode

        # used for statistics of invalid labels and data ids
        self.corrupt_idx = set()

        # create buffers
        self.length = round(len(labels) * self.fraction)
        self.create_buffers(self.length)

        # cache labels
        if isinstance(labels, np.ndarray):
            self.get_labels_np(labels)
        elif isinstance(labels, list):
            self.label_files = labels[: self.length]
            self.get_labels()
        else:
            raise TypeError(f"The input labels must be of np.ndarray or list type, but got {type(data)}.")
        self.length = len(self.labels)

        # cache data
        if isinstance(data, np.ndarray):
            self.cache_data_np(data)
        elif isinstance(data, list):
            self.data_files = data[: self.length]
            self.data_npy_files = [Path(f).with_suffix(".npy") for f in self.data_files]

            if self.cache == "ram" and self.check_cache_ram():
                self.cache_data()
            elif self.cache == "disk" and self.check_cache_disk():
                self.cache_data()
        else:
            raise TypeError(f"The input data must be of np.ndarray or list type, but got {type(data)}.")

        # update buffers length
        self.update_buffers()

        # Buffer thread for data fusion
        self.buffer = []
        self.max_buffer_length = min((self.length, self.batch_size * 8, 1000)) if self.augment else 0

        # Transforms
        self.transforms = self.build_transforms(hyp=hyp)

    def get_labels(self, desc: str = "") -> None:
        """Get labels from path list to buffers."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        LOGGER.info(f"Caching {self.mode} labels...")
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(self.cache_labels, range(len(self.label_files)))
            pbar = tqdm(enumerate(results), total=len(self.label_files))
            for _, lb in pbar:
                if lb:
                    b += asizeof.asizeof(lb)
                pbar.desc = f"Caching {self.mode} labels ({b / gb:.5f}GB)" + desc
            pbar.close()

    def get_labels_np(self, labels: np.ndarray) -> None:
        """Get labels from matrix input to buffers."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        LOGGER.info(f"Caching {self.mode} labels from matrix input...")
        pbar = tqdm(enumerate(labels), total=len(labels))
        for i, label in pbar:
            label = self.label_format(label)
            self.labels[i] = label
            b += asizeof.asizeof(label)
            pbar.desc = f"Caching {self.mode} labels ({b / gb:.5f}GB)"
        pbar.close()

    def cache_labels(self, i: int) -> dict[str, Any] | None:
        """Cache label from paths to ram for faster loading."""
        label = self.lread(i)
        label = self.label_format(label)
        if label:
            self.labels[i] = label
            return label
        else:  # label corrupt
            self.corrupt_idx.add(i)
            return None

    def cache_data_np(self, data: np.ndarray) -> None:
        """Cache np.ndarray format data to buffers"""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        LOGGER.info(f"Caching {self.mode} data from matrix input...")
        pbar = tqdm(enumerate(data), total=len(data))
        for i, data in pbar:
            self.data[i] = data
            b += data.nbytes
            pbar.desc = f"Caching {self.mode} data ({b / gb:.5f}GB)"
        pbar.close()

    def cache_data(self) -> None:
        """Cache data to memory or disk."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        fcn, storage = (self.load_data, "RAM") if self.cache == "ram" else (self.cache_data_to_disk, "Disk")
        LOGGER.info(f"Caching {self.mode} data to {storage}...")
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(len(self.data_files)))
            pbar = tqdm(enumerate(results), total=len(self.data_files))
            for i, x in pbar:
                if self.cache == "disk":
                    try:
                        b += self.data_files[i].stat().st_size
                    except FileNotFoundError:
                        b += 0.0
                else:
                    if x:
                        self.data[i] = x
                        b += self.data[i].nbytes
                pbar.desc = f"Caching {self.mode} data ({b / gb:.5f}GB {storage})"
            pbar.close()

    def cache_data_to_disk(self, i: int) -> float:
        """Cache data from paths to disk with as an .npy file for faster loading."""
        f = self.data_npy_files[i]
        if not f.exists():
            data = self.file_read(self.data_files[i])
            if data:
                np.save(f, data, allow_pickle=False)
            else:  # data corrupt
                self.corrupt_idx.add(i)

    def load_data(self, i: int) -> np.ndarray | None:
        """Loads 1 data and label from dataset index 'i', returns (data, label)."""
        dt, dtf, dtfn = self.data[i], self.data_files[i], self.data_npy_files[i]

        # load data
        if dt is None:  # not cached in RAM
            if dtfn.exists():  # cached in Disk
                try:
                    data = np.load(dtfn)
                except Exception as e:
                    LOGGER.warning(f"Removing corrupt *.npy image file {dtfn} due to: {e}")
                    Path(dtfn).unlink(missing_ok=True)
                    data = self.file_read(dtf)
            else:
                data = self.file_read(dtf)

            if data is not None:  # data corrupt
                # Add to buffer if training with augmentations
                if self.augment:
                    self.data[i] = data
                    self.buffer.append(i)
                    if len(self.buffer) >= self.max_buffer_length:  # prevent empty buffer
                        j = self.buffer.pop(0)
                        if self.cache != "ram":
                            self.data[j] = None
            else:
                self.corrupt_idx.add(i)

            return data

        return dt

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
        """Returns Data and label information for given index."""
        label = deepcopy(self.labels[index])
        data = self.load_data(index)
        label = self.update_labels_info(label)

        return data, label

    def update_labels_info(self, label: dict[str, Any]) -> dict[str, Any]:
        """Custom your label format here."""
        return label

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

    def lread(self, index: int) -> np.ndarray | None:
        """Users implement their own label reading logic."""
        label_path = self.label_files[index]
        label = self.file_read(label_path)

        return label

    @staticmethod
    def file_read(path: FilePath) -> np.ndarray | None:
        path = Path(path)
        extension = path.suffix.lower()

        if not path.exists():
            LOGGER.error(f"File does not exist: {path}")
            return None

        try:
            if extension in IMG_FORMATS:  # imgs
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
                    f"Unsupported file type: {extension}. Supported types: {list(IMG_FORMATS)} + ['.txt', '.npy']"
                )
                return None

        except Exception as e:
            LOGGER.error(f"Could not read file '{path}': {e}")
            return None

    def label_format(self, label: np.ndarray | None) -> dict[str, Any] | None:
        """format the label from np.ndarray to a custom form."""
        if label is not None:
            label = {"label": label}  # Customize
            return label
        else:
            return None

    def build_transforms(self, hyp: dict[str, Any] | None = None):
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

    def create_buffers(self, length: int) -> None:
        """Build buffers for data and labels storage."""
        self.labels = [None] * length
        self.label_files = [None] * length
        self.data = [None] * length
        self.data_files = [None] * length
        self.data_npy_files = [None] * length

    def update_buffers(self) -> None:
        """Update the buffer and delete invalid data (None)."""
        if self.corrupt_idx:
            LOGGER.info(f"Removing invalid items from buffers...: {[id for id in self.corrupt_idx]}")
            for i in self.corrupt_idx:
                self.remove_item(i)
        self.length = len(self.labels)

    def remove_item(self, i: int) -> None:
        """Remove a item of buffers according to the index."""
        # remove label
        self.labels.pop(i)
        self.label_files.pop(i)
        # remove data
        self.data.pop(i)
        self.data_files.pop(i)
        self.data_npy_files.pop(i)
