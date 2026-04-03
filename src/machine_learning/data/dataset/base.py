from typing import Any, Callable, Literal

import os
import re
import sys
import cv2
import torch
import random
import psutil
import warnings
import numpy as np

from tqdm import tqdm
from addict import Dict
from pathlib import Path
from copy import deepcopy
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from multiprocessing.pool import ThreadPool

from machine_learning.utils.logger import LOGGER
from machine_learning.utils.ops import is_empty_array
from machine_learning.utils.constants import NUM_THREADS, IMG_FORMATS, NPY_FORMATS


class DatasetBase(Dataset, ABC):
    """
    Base dataset class for data loading and processing.

    Adapted from Ultralytics YOLO base dataset implementation.
    Source: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/base.py

    Args:
        data (list[str] | np.ndarray): Path list to the data or data itself with np.ndarray format.
        labels (list[str] | np.ndarray): Path list to the labels or labels itself with np.ndarray format.
        cache (bool, optional): Cache data to RAM or disk during training. Defaults to False.
        augment (bool): If True, data augmentation is applied. Defaults to True.
        hyp (dict, optional): Hyperparameters for data augmentation. Defaults to None.
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
        _check_input_match(np.ndarray | list[str], np.ndarray | list[str]): Check whether the type and the lengths of
        the data and labels are consistent.
        create_buffers(int): Builds buffers for data and labels storage.
        setup_data_labels(np.ndarray | list[str], np.ndarray | list[str]): Setups labels and data storage. Labels are
        always cached, data is cached conditionally.
        cache_labels_np(np.ndarray): Caches labels of the dataset from np.ndarray input to buffers.
        cache_data_np(np.ndarray): Caches the data of the dataset from np.ndarray data source to buffers.
        cache_labels(Callable | None): Caches labels from label file paths to buffers for faster loading, the callback
        function input is used to count the cache information.
        get_labels(int): Reads labels from the specified path index and organize them into a specific format, internally
        calls label_read() and label_format() methods.
        label_read(int): Reads label from a specific path and verify the validity of relative data.
        label_format(Any | None): Formats the label to a custom dict form.
        cache_data(): Caches data to memory or disk.
        load_data(int): Loads 1 data from dataset index 'i'.
        cache_data_to_disk(int): Caches data from data_files index 'i' to Disk.
        __len__(): Returns the total number of samples in the dataset.
        __getitem__(int): Returns transformed sample for given index 'i'.
        get_sample(int): Returns a sample including data and annotations for given index 'i'.
        update_annotations(dict[str, Any]): Update annotations in a sample returned by get_sample() method.
        check_cache_ram(float): Checks data caching requirements vs available memory.
        check_cache_disk(float): Checks data caching requirements vs disk free space.
        build_transforms(dict[str, Any]): Builds data transform.
        register_buffer(str, list): Adds a buffer to the buffer dictionary to facilitate unified management of buffers.
        update_buffer(): Updates buffers and delete invalid items.
        remove_item(int): Removes a item of buffers according to the index 'i'.
        file_read(str): Static method that reads file content from given path.
        verify_data(str): Static method that validates whether the data from given path is vaild.
        normalize_cache(bool | Literal["ram", "disk"] | None): Standardized cache configuration options.
    """

    def __init__(
        self,
        data: np.ndarray | list[str],
        labels: np.ndarray | list[str],
        cache: bool | Literal["ram", "disk"] | None = False,
        augment: bool = False,
        hyp: dict[str, Any] = {},
        fraction: float = 1.0,
        mode: Literal["train", "val", "test"] = "train",
    ):
        """Initialize DatasetBase with given configuration and options."""
        super().__init__()

        self._check_input_match(data, labels)

        self.hyp = Dict(hyp)
        if not (0 < fraction <= 1.0):
            raise ValueError("Fraction must be in (0, 1].")
        self.fraction = fraction
        self.cache = self.normalize_cache(cache)
        self.mode = mode
        self.augment = augment

        # used for statistics of invalid labels and data ids
        self.corrupt_idx = set()

        # create buffers
        self.create_buffers(labels)

        # setup data and labels
        self.setup_data_labels(data, labels)

        # Transforms
        self.transforms = self.build_transforms(hyp=self.hyp)

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return self.length

    def _check_input_match(self, data: np.ndarray | list[str], labels: np.ndarray | list[str]):
        """
        Check whether the type and the lengths of the data and labels are consistent.
        """
        # empty check
        if len(data) == 0:
            raise ValueError("Data cannot be empty.")
        if len(labels) == 0:
            raise ValueError("Labels cannot be empty.")

        # type check
        if not isinstance(labels, (list, np.ndarray)):
            raise TypeError(f"Unsupported labels type: {type(labels)}. Expected list or np.ndarray.")
        if not isinstance(data, (list, np.ndarray)):
            raise TypeError(f"Unsupported data type: {type(data)}. Expected list or np.ndarray.")

        # length check
        if len(labels) != len(data):
            raise ValueError(f"Labels length {len(labels)} does not match data length {len(data)}.")

    def create_buffers(self, labels: np.ndarray | list[str]) -> None:
        """Build buffers for data and labels storage."""
        # calculate the length of the dataset after applying fraction
        self.length = round(len(labels) * self.fraction)

        # register buffers
        self.buffers = {}

        self.labels = [None] * self.length
        self.label_files = [None] * self.length
        self.data = [None] * self.length
        self.data_files = [None] * self.length
        self.data_npy_files = [None] * self.length

        self.buffers["labels"] = self.labels
        self.buffers["label_files"] = self.label_files
        self.buffers["data"] = self.data
        self.buffers["data_files"] = self.data_files
        self.buffers["data_npy_files"] = self.data_npy_files

    def setup_data_labels(self, data: np.ndarray | list[str], labels: np.ndarray | list[str]) -> None:
        """Setup labels and data storage. Labels are always cached, data is cached conditionally."""

        # ********************************************************
        #
        # add properties used for cache_data() here in subclasses.
        #
        # ********************************************************

        # While caching labels, the data path needs to be cached first to verify the validity.
        if isinstance(data, list):
            self.data_files[:] = data[: self.length]
            self.data_npy_files[:] = [Path(f).with_suffix(".npy") for f in self.data_files]

        # cache labels
        if isinstance(labels, np.ndarray):  # np.ndarray, mainly for classification labels
            self.cache_labels_np(labels)
        else:  # list[str], mainly for labels with file paths, such as detection labels
            self.label_files[:] = labels[: self.length]
            self.cache_labels()

        # cache data
        if isinstance(data, np.ndarray):  # np.ndarray, mainly for small data that can be parsed to np.ndarray directly
            self.cache_data_np(data)
        else:  # list[str], mainly for data with file paths, such as images, point clouds, etc.
            if self.cache == "ram" and self.check_cache_ram():
                self.cache_data()
            elif self.cache == "disk" and self.check_cache_disk():
                self.cache_data()

    def cache_labels_np(self, labels: np.ndarray) -> None:
        """Cache labels from np.ndarray input to buffers."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        LOGGER.info(f"Caching {self.mode} labels from np.ndarray input...")
        pbar = tqdm(enumerate(labels[: self.length]), total=self.length)
        for i, label in pbar:
            label = self.label_format(label)
            self.labels[i] = label
            b += self.calculate_cache_size(label)

            if i % 100 == 0 or i == len(pbar) - 1:  # Update the progress bar at intervals to reduce consumption.
                pbar.desc = f"Caching {self.mode} labels ({b / gb:.5f}GB)"

        pbar.close()

    def cache_data_np(self, data: np.ndarray) -> None:
        """Cache the data of the dataset from np.ndarray data source to buffers."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        LOGGER.info(f"Caching {self.mode} data from np.ndarray input...")
        pbar = tqdm(enumerate(data[: self.length]), total=self.length)
        for i, item in pbar:
            self.data[i] = item
            b += self.calculate_cache_size(item)

            if i % 100 == 0 or i == len(pbar) - 1:
                pbar.desc = f"Caching {self.mode} data ({b / gb:.5f}GB)"

        pbar.close()

    def cache_labels(self, desc_func: Callable | None = None) -> None:
        """Cache label from paths to ram for faster loading."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        LOGGER.info(f"Caching {self.mode} labels...")

        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(self.get_labels, range(self.length))
            pbar = tqdm(enumerate(results), total=self.length)
            for i, lb in pbar:
                if lb:
                    self.labels[i] = lb
                    b += self.calculate_cache_size(lb)
                else:  # label corrupt
                    self.corrupt_idx.add(i)

                if i % 100 == 0 or i == len(pbar) - 1:
                    pbar.desc = (
                        f"Caching {self.mode} labels ({b / gb:.5f}GB) " + desc_func() if desc_func is not None else ""
                    )
            pbar.close()

        # update buffers length
        self.update_buffers()

    def get_labels(self, i: int) -> dict[str, Any] | None:
        """Read labels from the specified path index and organize them into a specific format.

        Args:
            i (int): The index.

        Returns:
            dict: When label and data are valid, return label in a customized dict interface, otherwise return None.
        """
        label = self.label_read(i)
        label = self.label_format(label)

        return label

    @abstractmethod
    def label_read(self, i: int) -> Any:
        """
        Read label from a specific path and verify the validity of relative data.

        Example implementation:
            data_file, lb_file = self.data_files[i], self.label_files[i]
            try:
                if not self.verify_data(data_file):
                    LOGGER.warning(f"Invalid data file: {data_file}")
                    return None
                label = self.file_read(lb_file)
            except Exception as e:
                LOGGER.error(f"Error reading label at index {i}: {e}")
                return None
            return label
        """
        raise NotImplementedError("Subclasses must implement label_read() method.")

    @abstractmethod
    def label_format(self, label: Any) -> dict[str, Any] | None:
        """Format the label to a custom dict form."""
        raise NotImplementedError("Subclasses must implement label_format() method.")

    def cache_data(self) -> None:
        """Cache data to memory or disk."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        fcn, storage = (self.load_data, "RAM") if self.cache == "ram" else (self.cache_data_to_disk, "Disk")
        LOGGER.info(f"Caching {self.mode} data to {storage}...")

        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(self.length))
            pbar = tqdm(enumerate(results), total=self.length)
            for i, x in pbar:
                if self.cache == "disk":
                    b += x
                else:
                    if x is not None:
                        self.data[i] = x
                        b += self.calculate_cache_size(self.data[i])
                if i % 100 == 0 or i == len(pbar) - 1:
                    pbar.desc = f"Caching {self.mode} data ({b / gb:.5f}GB {storage})"

            pbar.close()

    def cache_data_to_disk(self, i: int) -> float:
        """Cache data from paths to disk with as an .npy file for faster loading and return its size."""
        f = self.data_npy_files[i]
        if not f.exists():
            data = self.file_read(self.data_files[i])
            if data is not None:
                np.save(f, data, allow_pickle=False)
                return f.stat().st_size
            else:
                return 0.0
        else:
            # Even if the cache already exists, the actual size should be returned to ensure accurate statistics.
            return f.stat().st_size

    def load_data(self, i: int) -> np.ndarray | None:
        """Loads 1 data and label from dataset index 'i', returns (data, label)."""
        dt, dtf, dtfn = self.data[i], self.data_files[i], self.data_npy_files[i]

        # load data
        if dt is None:  # not cached in RAM
            if dtfn.exists():  # cached in Disk
                try:
                    dt = np.load(dtfn)
                except Exception as e:
                    LOGGER.warning(f"Removing corrupt *.npy image file {dtfn} due to: {e}")
                    Path(dtfn).unlink(missing_ok=True)
                    dt = self.file_read(dtf)
            else:
                dt = self.file_read(dtf)

            return dt

        return dt

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Returns transformed sample for given index 'i'."""
        sample = self.get_sample(index)
        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def get_sample(self, index: int) -> dict[str, Any]:
        """Returns a sample including data and annotations for given index 'i'."""
        sample = deepcopy(self.labels[index])
        sample["data"] = self.load_data(index)
        sample = self.update_annotations(sample)

        return sample

    def update_annotations(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Update your annotations here."""
        return sample

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
            b += self.calculate_cache_size(data)

        valid_samples = n - skips
        if valid_samples == 0:
            self.cache = None
            LOGGER.warning("All sampled data failed to load. Disabling RAM cache.")
            return False

        mem_required = b * self.length / valid_samples * (1 + safety_margin)  # GB required to cache data into RAM
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

        target_dir = Path(self.data_npy_files[0]).parent
        if not os.access(target_dir, os.W_OK):
            self.cache = None
            LOGGER.info(f"Skipping caching data to disk, directory {target_dir} is not writeable.")
            return False

        b, gb = 0, 1 << 30  # bytes of cached data, bytes per gigabytes
        n = min(self.length, 100)  # extrapolate from 30 random data
        skips = 0

        for _ in range(n):
            i = random.randint(0, self.length - 1)
            data = self.file_read(self.data_files[i])
            if data is None:
                skips += 1
                continue
            b += self.calculate_cache_size(data)

        valid_samples = n - skips
        if valid_samples == 0:
            self.cache = None
            LOGGER.warning("All sampled data failed to load. Disabling Disk cache.")
            return False

        disk_required = b * self.length / valid_samples * (1 + safety_margin)  # bytes required to cache data to disk
        total, _, free = shutil.disk_usage(target_dir)

        if disk_required > free:
            self.cache = None
            LOGGER.info(
                f"{disk_required / gb:.1f}GB disk space required, "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{free / gb:.1f}/{total / gb:.1f}GB free, not caching data to disk."
            )
            return False

        return True

    @abstractmethod
    def build_transforms(self, hyp: dict[str, Any] | None = None):
        """
        Users can customize augmentations here.

        Example implementation:

            if self.augment:
                return Compose([])
            else:
                return Compose([])
        """
        raise NotImplementedError("Subclasses must implement build_transforms() method.")

    def register_buffer(self, name: str, buffer: list) -> None:
        """Add a buffer to the buffer dictionary to facilitate unified management of buffers."""
        if len(buffer) != self.length:
            raise ValueError(f"The length of {name} buffer must be equal to length of the dataset.")

        if name not in self.buffers.keys():
            self.buffers[name] = buffer
        else:
            raise KeyError(f"Buffer {name} already exists.")

    def update_buffers(self) -> None:
        """Update the buffer and delete invalid items."""
        if self.corrupt_idx:
            LOGGER.info(f"Removing invalid items from buffers...: {list(self.corrupt_idx)}")

            for i in sorted(self.corrupt_idx, reverse=True):
                self.remove_item(i)

            LOGGER.info(f"Updated buffers length: {self.length}")
            self.corrupt_idx.clear()

    def remove_item(self, i: int) -> None:
        """Remove a item of buffers according to the index 'i'."""
        for buffer in self.buffers.values():
            buffer.pop(i)
        self.length = len(self.labels)

    @staticmethod
    def file_read(data_file: str) -> np.ndarray | None:
        """
        Read data file based on file extension.

        Args:
            data_file: Path to the data file.

        Returns:
            np.ndarray: Ndarray if file is accessible, None otherwise.
        """

        # get file extension
        path = Path(data_file)
        extension = path.suffix.lower()

        if not path.exists():
            LOGGER.error(f"File does not exist: {path}.")
            return None
        elif not path.is_file():
            LOGGER.error(f"The path is not a file: {path}.")
            return None

        try:
            # read img file with cv2
            if extension in IMG_FORMATS:  # imgs
                data = cv2.imread(str(path))  # bgr
                # data = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)  # bgr
                if data is None:
                    LOGGER.error(f"Failed to read image: {path}")
                    return None
                return data

            elif extension == ".txt":  # text
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    data = np.loadtxt(path, dtype=np.float32)
                    if is_empty_array(data):
                        LOGGER.warning(f"Empty text file: {path}")
                        return None
                return data

            elif extension == ".npy":  # numpy
                data = np.load(path)
                if is_empty_array(data):
                    LOGGER.warning(f"Empty numpy file: {path}")
                    return None
                return data

            else:
                LOGGER.error(
                    f"Unsupported file type: {extension}. Supported types: {list(IMG_FORMATS)} + ['.txt', '.npy']"
                )
                return None

        except Exception as e:
            LOGGER.error(f"Could not read file '{path}': {e}")
            return None

    @staticmethod
    def verify_data(data_file: str) -> bool:
        """
        Quickly verify data file availability based on file extension.

        Args:
            data_file: Path to the data file.

        Returns:
            bool: True if file is likely valid and accessible, False otherwise.
        """
        try:
            file_path = Path(data_file)

            # Check if file exists and is accessible
            if not file_path.exists() or not file_path.is_file():
                return False

            # Check file size (basic sanity check)
            if file_path.stat().st_size == 0:
                return False

            # Extension-based validation
            extension = file_path.suffix.lower()

            if extension in IMG_FORMATS:
                return _verify_image(file_path)
            elif extension in NPY_FORMATS:
                return _verify_numpy(file_path)
            elif extension in {".txt", ".csv"}:
                return _verify_text(file_path)
            elif extension in {".json", ".xml"}:
                return _verify_structured(file_path)
            elif extension in {".pkl", ".pickle"}:
                return _verify_pickle(file_path)
            else:
                # For unknown extensions, do basic file check
                return _verify_generic(file_path)

        except (OSError, IOError, PermissionError):
            return False

    @staticmethod
    def normalize_cache(cache: bool | Literal["ram", "disk"] | None) -> Literal["ram", "disk"] | None:
        """Standardized cache configuration options."""
        if cache is True:
            return "ram"
        if cache is False or cache is None:
            return None
        if isinstance(cache, str):
            cache_l = cache.lower()
            if cache_l in ("ram", "disk"):
                return cache_l
        raise ValueError("Cache must be True/False/'ram'/'disk'/None")

    @staticmethod
    def calculate_cache_size(data: Any, _seen: set = None) -> int:
        """Calculate the exact size of data to be cached in bytes, preventing double-counting."""
        if _seen is None:
            _seen = set()

        obj_id = id(data)
        if obj_id in _seen:
            return 0  # Avoid circular references and redundant calculations in shared memory (View)
        _seen.add(obj_id)

        if data is None:
            return 0

        if isinstance(data, torch.Tensor):
            return sys.getsizeof(data) + data.element_size() * data.nelement()

        if isinstance(data, np.ndarray):
            size = sys.getsizeof(data)
            if data.base is not None:
                size += DatasetBase.calculate_cache_size(data.base, _seen)

            return size

        if isinstance(data, dict):
            total = sys.getsizeof(data)
            for k, v in data.items():
                total += DatasetBase.calculate_cache_size(k, _seen)
                total += DatasetBase.calculate_cache_size(v, _seen)
            return total

        if isinstance(data, (list, tuple, set)):
            total = sys.getsizeof(data)
            for item in data:
                total += DatasetBase.calculate_cache_size(item, _seen)
            return total

        # Attempting to iterate over the __dict__ of a custom object.
        if hasattr(data, "__dict__"):
            total = sys.getsizeof(data)
            for attr_name, attr_value in data.__dict__.items():
                total += DatasetBase.calculate_cache_size(attr_name, _seen)
                total += DatasetBase.calculate_cache_size(attr_value, _seen)
            return total

        return sys.getsizeof(data)


class MultiModalDatasetBase(DatasetBase):
    """
    Base multimodal dataset class for loading and processing multimodal data and labels.

    Args:
        data (dict[str, np.ndarray | list[str]]): Multimodal data path list or data arrays.
        labels (np.ndarray | list[str] | dict[str, list[str]]): Label path lists or label arrays.
        cache (bool, optional): Whether to cache data to RAM or disk during training. Defaults to False.
        augment (bool, optional): If True, data augmentation is applied. Defaults to True.
        hyp (dict, optional): Hyperparameters for data augmentation. Defaults to None.
        fraction (float): Fraction of dataset to use. Defaults to 1.0 (use all data).
        modals (list[str] | None): List of modal names. Defaults to None.
        dropout (bool): Whether some modality data is allowed to be missing.
        mode (Literal["train", "val", "test"]): Dataset mode.

    Properties:
        modals (list[str]): List of multimodal data names.
        label_names (list[str] | None): List of multimodal label names.
        data (dict[str, list[np.ndarray]]): Multimodal dict-buffer contains list-buffers for each modality data.
        data_files (dict[str, list[str]]): Multimodal dict-buffer contains list-buffers for each modality data paths.
        data_npy_files (dict[str, list[str]]): Multimodal dict-buffer has list-buffers for each modality .npy files.
        labels (list[dict[str, Any]]): The list-buffer for label data.
        label_files (list[str] | dict[str, list[str]]): The list-buffer or list-buffers dict for multimodal label files.
        length (int): Number of data samples in the dataset.
        transforms (callable): Transformation function.
        modal_mapping: Mapping dict that map the plural form of data names to singular modal names.

    Methods:
        _check_input_match(): Check whether the lengths of the multimodal data and labels are consistent, and whether
        the types and structures match appropriately.
        create_buffers(): Build buffers for storage.
        setup_data_labels(): Setups labels and data storage. Labels are always cached, data is cached conditionally.
        cache_labels_np(): Cache labels from matrix input to buffers.
        cache_labels(): Cache labels from path list to buffers.
        get_labels(): Reads labels from the specified path index 'i' and organize them into a specific format.
        label_read(): Read label from a specific path and verify the validity of relative data.
        label_format(): Format the label to a custom dict form.
        cache_data_np(): Cache np.ndarray format data to buffers.
        cache_data(): Cache data to memory or disk.
        load_data(): Loads 1 multimodal data from dataset index 'i', and return a data dict.
        cache_data_to_disk(): Cache data from paths to disk with as an .npy file for faster loading.
        __len__(): Returns the length of the dataset.
        __getitem__(): Returns transformed sample for given index.
        get_sample(): Returns a sample for given index.
        update_annotations(): Update your annotations here.
        check_cache_ram(): Check data caching requirements vs available memory.
        check_cache_disk(): Check data caching requirements vs disk free space.
        build_transforms(): Build data transformation pipeline.
        register_buffer(): Add a buffer to the buffer management dict to facilitate unified management of buffers.
        update_buffers(): Update the buffer and delete invalid items.
        remove_item(): Remove a item of buffers according to the index.
        to_singular(): Intelligence converts plural modal name into singular forms.
        modal_filter(): Map the plural form of data names to singular modal names.
    """

    def __init__(
        self,
        data: dict[str, np.ndarray | list[str]],
        labels: np.ndarray | list[str] | dict[str, list[str] | np.ndarray],
        cache: bool | Literal["ram", "disk"] | None = False,
        augment: bool = False,
        hyp: dict[str, Any] = {},
        fraction: float = 1.0,
        modalities: list[str] | None = None,
        dropout: bool = False,
        mode: Literal["train", "val", "test"] = "train",
    ):
        """Initialize MultiModalDatasetBase with given configuration and options."""

        self.dropout = dropout
        self._data_names = list(data.keys())  # used for building buffers
        self._label_names = list(labels.keys()) if isinstance(labels, dict) else None

        if len(self._data_names) < 2:
            raise ValueError("The data modalities must be greater than or equal to 2.")

        # deal with modal names
        if modalities is not None:
            if len(modalities) != len(self._data_names):
                raise ValueError(
                    f"The length of modalities ({len(modalities)}) must match data keys ({len(self._data_names)})"
                )
            self._modalities = modalities
        else:
            self._modalities = [self.to_singular(name) for name in self._data_names]  # imgs->img, depths->depth, etc.
        self.modality_maps = {
            data_name: self.modality_filter(data_name) for data_name in self._data_names
        }  # images: img

        super().__init__(
            data=data,
            labels=labels,
            cache=cache,
            augment=augment,
            hyp=hyp,
            fraction=fraction,
            mode=mode,
        )

    @property
    def data_names(self) -> list[str]:
        return self._data_names

    @property
    def modalities(self) -> list[str]:
        return self._modalities

    @property
    def label_names(self) -> list[str] | None:
        return self._label_names

    def _check_input_match(
        self,
        data: dict[str, np.ndarray | list[str]],
        labels: np.ndarray | list[str] | dict[str, list[str] | np.ndarray],
    ):
        """
        Check whether the lengths of the multimodal data and labels are consistent, and whether the types and structures
        match appropriately.
        """
        # empty check
        if not data:
            raise ValueError("Data dictionary cannot be empty.")

        # data length check
        data_lens = {name: len(v) for name, v in data.items()}
        if 0 in data_lens.values():
            raise ValueError(f"Data entries cannot be empty, got: {data_lens}.")  # empty check for each data entry
        if len(set(data_lens.values())) != 1:
            raise ValueError(f"All data must have the same length, got: {data_lens}.")
        data_len = next(iter(data_lens.values()))

        # data type check
        data_types = {type(v) for v in data.values()}
        if len(data_types) != 1:
            raise ValueError(f"All data values must be of the same type, got: {data_types}.")

        # labels type and length check
        if isinstance(labels, (list, np.ndarray)):
            if len(labels) == 0:
                raise ValueError("Labels cannot be empty.")

            if len(labels) != data_len:
                raise ValueError(f"Labels length {len(labels)} not equal to data length {data_len}.")

        elif isinstance(labels, dict):
            if not labels:
                raise ValueError("Labels dictionary cannot be empty.")

            label_lens = {k: len(v) for k, v in labels.items()}
            if 0 in label_lens.values():
                raise ValueError(f"Label entries cannot be empty, got: {label_lens}.")
            if len(set(label_lens.values())) != 1:
                raise ValueError(f"All label entries must have the same length, got: {label_lens}.")
            label_len = next(iter(label_lens.values()))
            if label_len != data_len:
                raise ValueError(f"Labels length {label_len} != data length {data_len}.")

            label_types = {type(v) for v in labels.values()}
            if len(label_types) != 1:
                raise ValueError(f"All label values must be of the same type, got: {label_types}.")

        else:
            raise TypeError(f"Unsupported labels type: {type(labels)}.")

    def create_buffers(self, labels: np.ndarray | list[str] | dict[str, list[str] | np.ndarray]) -> None:
        """Build buffers for data and labels storage."""
        # calculate the length of the dataset after applying fraction
        if isinstance(labels, (np.ndarray, list)):
            length = len(labels)
        elif isinstance(labels, dict):
            length = len(next(iter(labels.values())))
        self.length = round(length * self.fraction)

        self.buffers = {}

        self.data = {name: [None] * self.length for name in self.data_names}
        self.data_files = {name: [None] * self.length for name in self.data_names}
        self.data_npy_files = {name: [None] * self.length for name in self.data_names}
        self.labels = [None] * self.length  # labels buffers

        self.buffers["data"] = self.data
        self.buffers["data_files"] = self.data_files
        self.buffers["data_npy_files"] = self.data_npy_files
        self.buffers["labels"] = self.labels

        if self.label_names is not None:
            if isinstance(next(iter(labels.values())), list):
                self.label_files = {name: [None] * self.length for name in self.label_names}
                self.buffers["label_files"] = self.label_files
        elif isinstance(labels, list):
            self.label_files = [None] * self.length
            self.buffers["label_files"] = self.label_files

    def setup_data_labels(
        self,
        data: dict[str, np.ndarray | list[str]],
        labels: np.ndarray | list[str] | dict[str, np.ndarray | list[str]],
    ) -> None:
        """Setups labels and data storage. Labels are always cached, data is cached conditionally."""

        # ********************************************************
        #
        # add properties used for cache_data() here in subclasses.
        #
        # ********************************************************

        # While caching labels, the data path needs to be cached first to verify the validity.
        if isinstance(next(iter(data.values())), list):
            for name in self.data_names:
                self.data_files[name][:] = data[name][: self.length]
                self.data_npy_files[name][:] = [Path(f).with_suffix(".npy") for f in self.data_files[name]]

        # cache labels
        if isinstance(labels, np.ndarray) or (
            isinstance(labels, dict) and isinstance(next(iter(labels.values())), np.ndarray)
        ):  # np.ndarray or dict[str, np.ndarray]
            self.cache_labels_np(labels)
        else:  # list[str] or dict[str, list[str]]
            if isinstance(labels, list):
                self.label_files[:] = labels[: self.length]
            elif isinstance(labels, dict):
                for name in self.label_names:
                    self.label_files[name][:] = labels[name][: self.length]
            self.cache_labels()

        # cache data
        if isinstance(next(iter(data.values())), np.ndarray):  # dict[str, np.ndarray]
            self.cache_data_np(data)
        else:  # dict[str, list[str]]
            if self.cache == "ram" and self.check_cache_ram():
                self.cache_data()
            elif self.cache == "disk" and self.check_cache_disk():
                self.cache_data()

    def cache_labels_np(self, labels: np.ndarray | dict[str, np.ndarray]) -> None:
        """Cache labels from np.array input to buffers."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        LOGGER.info(f"Caching {self.mode} labels from np.array input...")
        if isinstance(labels, np.ndarray):
            pbar = tqdm(enumerate(labels[: self.length]), total=self.length)
            for i, label in pbar:
                label = self.label_format(label)
                self.labels[i] = label
                b += self.calculate_cache_size(label)

                if i % 100 == 0 or i == len(pbar) - 1:  # Update the progress bar at intervals to reduce consumption.
                    pbar.desc = f"Caching {self.mode} labels ({b / gb:.5f}GB)"

        else:
            pbar = tqdm(range(self.length))
            for i in pbar:
                lb = {}  # multi labels, e.g. {"img": [], "ir": []}
                for name in self.label_names:
                    lb[self.modality_maps[name]] = labels[name][i]
                lb = self.label_format(lb)
                self.labels[i] = lb
                b += self.calculate_cache_size(lb)

                if i % 100 == 0 or i == len(pbar) - 1:  # Update the progress bar at intervals to reduce consumption.
                    pbar.desc = f"Caching {self.mode} labels ({b / gb:.5f}GB)"

        pbar.close()

    def cache_data_np(self, data: dict[str, np.ndarray]) -> None:
        """Cache np.ndarray format data to buffers."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        LOGGER.info(f"Caching {self.mode} data from matrix input...")
        pbar = tqdm(range(self.length))
        for i in pbar:
            for name in self.data_names:
                self.data[name][i] = data[name][i]
                b += self.calculate_cache_size(self.data[name][i])

            if i % 100 == 0 or i == len(pbar) - 1:  # Update the progress bar at intervals to reduce consumption.
                pbar.desc = f"Caching {self.mode} data ({b / gb:.5f}GB)"

        pbar.close()

    def cache_data(self) -> None:
        """Cache data to memory or disk."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        fcn, storage = (self.load_data, "RAM") if self.cache == "ram" else (self.cache_data_to_disk, "Disk")
        LOGGER.info(f"Caching {self.mode} data to {storage}...")

        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(self.length))
            pbar = tqdm(enumerate(results), total=self.length)
            for i, x in pbar:
                if self.cache == "disk":
                    b += x
                else:
                    if x:
                        for name in self.data_names:
                            self.data[name][i] = x[self.modality_maps[name]]
                            b += self.calculate_cache_size(self.data[name][i])

                if i % 100 == 0 or i == len(pbar) - 1:
                    pbar.desc = f"Caching {self.mode} data ({b / gb:.5f}GB {storage})"

            pbar.close()

    def cache_data_to_disk(self, i: int) -> float:
        """Cache data from paths to disk with as an .npy file for faster loading."""
        b = 0.0
        for name in self.data_names:
            f: Path = self.data_npy_files[name][i]
            if not f.exists():
                data = self.file_read(self.data_files[name][i])
                if data is not None:
                    np.save(f, data, allow_pickle=False)
                    b += f.stat().st_size
            else:
                b += f.stat().st_size  # Even exists, the actual size should be returned to ensure accurate statistics.
        return b

    def load_data(self, i: int) -> dict[str, np.ndarray]:
        """Loads multimodal data from dataset index 'i', and return a data dict."""
        data = {}

        for name in self.data_names:
            dt, dtf, dtfn = self.data[name][i], self.data_files[name][i], self.data_npy_files[name][i]

            # load data
            if dt is None:  # not cached in RAM
                if dtfn.exists():  # cached in Disk
                    try:
                        dt = np.load(dtfn)
                    except Exception as e:
                        LOGGER.warning(f"Removing corrupt *.npy image file {dtfn} due to: {e}")
                        Path(dtfn).unlink(missing_ok=True)
                        dt = DatasetBase.file_read(dtf)
                else:
                    dt = DatasetBase.file_read(dtf)

                if dt is None:  # modal dropout
                    data[self.modality_maps[name]] = np.array([])
                else:
                    data[self.modality_maps[name]] = dt

            else:
                data[self.modality_maps[name]] = dt

        return data

    def get_sample(self, index: int) -> dict[str, Any]:
        """Returns a sample for given index."""
        sample = deepcopy(self.labels[index])
        sample.update(self.load_data(index))
        sample = self.update_annotations(sample)

        return sample

    def check_cache_ram(self, safety_margin: float = 0.5) -> bool:
        """Check data caching requirements vs available memory."""
        b, gb = 0, 1 << 30  # bytes of cached data, bytes per gigabytes
        n = min(self.length, 100)  # extrapolate from 100 random data
        skips = 0

        for _ in range(n):
            i = random.randint(0, self.length - 1)
            data = []
            for name in self.data_names:
                dt = DatasetBase.file_read(self.data_files[name][i])
                if dt is None:  # Strictly calculate and estimate the maximum value
                    data = []
                    skips += 1
                    break
                else:
                    data.append(dt)
            b += np.sum([self.calculate_cache_size(d) for d in data])

        valid_samples = n - skips
        if valid_samples == 0:
            self.cache = None
            LOGGER.warning("All sampled data failed to load. Disabling RAM cache.")
            return False

        mem_required = b * self.length / valid_samples * (1 + safety_margin)  # GB required to cache data into RAM
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

        for name in self.data_names:
            # check I/O access
            if not os.access(Path(self.data_files[name][0]).parent, os.W_OK):
                self.cache = None
                LOGGER.info("Skipping caching data to disk, directory not writeable.")
                return False

        for _ in range(n):
            i = random.randint(0, self.length - 1)
            data = []
            for name in self.data_names:
                # read data
                dt = DatasetBase.file_read(self.data_files[name][i])
                if dt is None:  # Strictly calculate and estimate the maximum value
                    skips += 1
                    data = []
                    break
                else:
                    data.append(dt)
            b += np.sum([self.calculate_cache_size(d) for d in data])

        valid_samples = n - skips
        if valid_samples == 0:
            self.cache = None
            LOGGER.warning("All sampled data failed to load. Disabling Disk cache.")
            return False

        disk_required = b * self.length / valid_samples * (1 + safety_margin)  # bytes required to cache data to disk
        sample_path = self.data_files[next(iter(self.data_names))][0]
        mount_point = get_mount_point(sample_path)
        total, _, free = shutil.disk_usage(mount_point)

        if disk_required > free:
            self.cache = None
            LOGGER.info(
                f"{disk_required / gb:.1f}GB disk space required, "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{free / gb:.1f}/{total / gb:.1f}GB free, not caching data to disk."
            )
            return False

        return True

    def register_buffer(self, name: str, buffer: list | dict[str, list]) -> None:
        """Add a buffer item to the buffer management dictionary to facilitate unified management of buffers."""
        if isinstance(buffer, list):
            if len(buffer) != self.length:
                raise ValueError(f"The length of {name} buffer must be equal to length of the dataset.")

        elif isinstance(buffer, dict):
            if set(buffer.keys()) != set(self.data_names):
                raise ValueError(f"The keys of {name} buffer must be the same as data buffer.")
            for sub in buffer.values():
                if not isinstance(sub, list):
                    raise TypeError("Each sub-buffer must be a list.")
                if len(sub) != self.length:
                    raise ValueError(f"The length of {name} buffer must be equal to length of the dataset.")

        if name not in self.buffers.keys():
            self.buffers[name] = buffer
        else:
            raise KeyError(f"Buffer {name} already exists.")

    def remove_item(self, i: int) -> None:
        """Remove a item of buffers according to the index."""
        for buffer in self.buffers.values():
            if isinstance(buffer, list):
                buffer.pop(i)
            elif isinstance(buffer, dict):
                for sub_buffer in buffer.values():
                    sub_buffer.pop(i)

        self.length = len(self.labels)

    def to_singular(self, string: str) -> str:
        "Intelligence converts plural modal name into singular forms."
        # Common rules for converting plurals to singulars
        plural_to_singular = {
            "ies": "y",  # cities -> city
            "ses": "s",  # buses -> bus
            "xes": "x",  # boxes -> box
            "zes": "z",  # quizzes -> quiz
            "ches": "ch",  # churches -> church
            "shes": "sh",  # dishes -> dish
            "s": "",  # cats -> cat
        }

        # Apply convert rules
        for plural, singular in plural_to_singular.items():
            if string.endswith(plural):
                return string[: -len(plural)] + singular

        # if no match, return raw string
        return string

    def modality_filter(self, string: str) -> str:
        """Map the plural form of data names to singular modal names."""
        for m in self.modalities:
            if re.match(m, string):
                return m


"""
Hepler function
"""


def _verify_image(file_path: Path) -> bool:
    """Verify image file using pillow."""
    try:
        from PIL import Image

        img = Image.open(file_path)
        img.verify()  # PIL verify
        if "." + img.format.lower() not in IMG_FORMATS:
            raise TypeError(f"Invalid image format {img.format}.")
        return True

    except Exception:
        return False


def _verify_numpy(file_path: Path) -> bool:
    """Verify numpy file integrity."""
    try:
        import numpy as np

        with open(file_path, "rb") as f:
            # Quick header check without full load
            version = np.lib.format.read_magic(f)
            return version is not None

    except Exception:
        return False


def _verify_text(file_path: Path) -> bool:
    """Verify text file readability."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            # Try to read first few bytes
            f.read(1024)
        return True

    except Exception:
        return False


def _verify_structured(file_path: Path) -> bool:
    """Verify JSON/XML file structure."""
    try:
        import json

        with open(file_path, "r", encoding="utf-8") as f:
            if file_path.suffix.lower() == ".json":
                json.load(f)  # Basic syntax check
            # For XML, we just check if it's readable
            content = f.read()
            return len(content) > 0

    except Exception:
        return False


def _verify_pickle(file_path: Path) -> bool:
    """Verify pickle file integrity."""
    try:
        import pickle

        with open(file_path, "rb") as f:
            pickle.load(f)  # Try to load
        return True

    except Exception:
        return False


def _verify_generic(file_path: Path) -> bool:
    """Generic file verification."""
    try:
        with open(file_path, "rb") as f:
            f.read(1)  # Try to read first byte
        return True

    except Exception:
        return False


def get_mount_point(path: Path) -> str:
    path = os.path.abspath(path)
    while not os.path.ismount(path):
        parent = os.path.dirname(path)
        if parent == path:  # Linux and Windows
            break
        path = parent
    return path
