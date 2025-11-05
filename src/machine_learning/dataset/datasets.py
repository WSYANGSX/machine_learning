from typing import Any, Optional, Literal

import os
import cv2
import math
import torch
import random
import psutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from itertools import chain
from multiprocessing.pool import ThreadPool

from machine_learning.utils.logger import LOGGER
from machine_learning.types.aliases import FilePath
from machine_learning.utils.constants import IMG_FORMATS, NUM_THREADS
from machine_learning.dataset.base import DatasetBase, MultiModalDatasetBase
from machine_learning.utils.transforms import Compose, Format, Instances, LetterBox, v8_transforms

from ultralytics.utils.ops import segments2boxes
from ultralytics.utils.ops import resample_segments


class ClassificationDataset(DatasetBase):
    """
    Dataset class for loading data and labels for classification. Based from DatasetBase class.

    Args:
        data (list[str] | np.ndarray): Path list to the data or data itself with np.ndarray format.
        labels (list[str] | np.ndarray): Path list to the labels or labels itself with np.ndarray format.
        batch_size (int, optional): Size of batches.
        cache (bool, optional): Cache data to RAM or disk during training. Defaults to False.
        augment (bool, optional): If True, data augmentation is applied. Defaults to True.
        hyp (dict, optional): Hyperparameters to apply data augmentation. Defaults to None.
        fraction (float): Fraction of dataset to utilize. Default is 1.0 (use all data).
        mode (Literal["train", "val", "test"]): The mode of the dataset.

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an classification model.
    """

    def __init__(
        self,
        data: np.ndarray | list[str],
        labels: np.ndarray | list[str],
        batch_size: int,
        cache: bool | Literal["ram", "disk"] | None = False,
        augment: bool = True,
        hyp: dict[str, Any] = {},
        fraction: float = 1.0,
        mode: Literal["train", "val", "test"] = "train",
    ):
        super().__init__(
            data=data,
            labels=labels,
            batch_size=batch_size,
            cache=cache,
            augment=augment,
            hyp=hyp,
            fraction=fraction,
            mode=mode,
        )

    def label_format(self, label: Any | None) -> dict[str, Any] | None:
        """Format the label to a dict with 'cls' item."""
        if label is not None and not isinstance(label, dict):
            label = {"cls": label}  # customize dict interface
            return label
        else:
            return label


class YoloDataset(DatasetBase):
    """
    Dataset class for loading images and labels for object detection and/or segmentation tasks in YOLO format.

    Args:
        imgs (list[str]): Path list to imgs or imgs itself with np.ndarray format.
        labels (list[str]): Path list to the labels or labels itself with np.ndarray format.
        imgsz (int): Image size. Defaults to 640.
        nc (int): Num of classes. Defaults to 80.
        rect (bool): If True, rectangular training is used. Defaults to False.
        stride (int): Stride. Defaults to 32.
        pad (float): Padding. Defaults to 0.0.
        single_cls (bool): If True, single class training is used. Defaults to False.
        classes (list): List of included classes. Default is None.
        cache (bool): Cache data to RAM or disk during training. Defaults to False.
        augment (bool): If True, data augmentation is applied. Defaults to True.
        hyp (dict, optional): Hyperparameters to apply data augmentation. Defaults to None.
        batch_size (int): Size of batches. Defaults to None.
        fraction (float): Fraction of dataset to utilize. Default is 1.0 (use all data).
        task ("detect", "pose", "segment"): The task of this dataset used for. Defaults to "detect".
        mode (Literal["train", "val", "test"]): The mode of the dataset.

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an detection or segmentation
        model.
    """

    def __init__(
        self,
        imgs: np.ndarray | list[FilePath],
        labels: np.ndarray | list[FilePath],
        imgsz: int = 640,
        nc: int = 80,
        rect: bool = False,
        stride: int = 32,
        pad: float = 0.5,
        single_cls: bool = False,
        classes: list[int] = None,
        cache: bool = False,
        augment: bool = True,
        hyp: dict[str, Any] | None = None,
        batch_size: int = 16,
        fraction: float = 1.0,
        task: Literal["detect", "pose", "segment", "obb"] = "detect",
        mode: Literal["train", "val", "test"] = "train",
    ):
        """Initialize BaseDataset with given configuration and options."""
        self.imgsz = imgsz
        self.rect = rect
        self.stride = stride
        self.pad = pad
        self.single_cls = single_cls
        self.classes = classes
        self.nc = nc
        self.batch_size = batch_size

        self.task = task
        self.use_segments = self.task == "segment"
        self.use_keypoints = self.task == "pose"
        self.use_obb = self.task == "obb"

        super().__init__(
            data=imgs,
            labels=labels,
            cache=cache,
            batch_size=batch_size,
            augment=augment,
            hyp=hyp,
            fraction=fraction,
            mode=mode,
        )  # cache imgs and labels

    @property
    def imgs(self) -> list[np.ndarray | str]:
        return self.data

    @property
    def img_files(self) -> list[str]:
        return self.data_files

    @property
    def img_npy_files(self) -> list[str]:
        return self.data_npy_files

    @imgs.setter
    def imgs(self, value: list[np.ndarray | str]) -> None:
        self.data = value

    @img_files.setter
    def img_files(self, value: list[str]) -> None:
        self.data_files = value

    @img_npy_files.setter
    def img_npy_files(self, value: list[str]) -> None:
        self.data_npy_files = value

    def setup_data_labels(self, data, labels):
        """Setup labels and data storage. Labels are always cached, data is cached conditionally."""
        # add mosaic_buffer used for images fusion
        self.mosaic_buffer = []
        self.max_mosaic_buffer_length = min((self.length, self.batch_size * 8, 1000)) if self.augment else 0

        super().setup_data_labels(data, labels)

    def cache_labels(self):
        """Get labels from path list to buffers."""
        # statistical indicators
        self.num_missing = 0
        self.num_found = 0
        self.num_empty = 0
        self.num_corrupt = 0

        # key points setting
        self.nkpt, self.kpt_dim = self.hyp.get("kpt_shape", (0, 0))
        if self.use_keypoints and (self.nkpt <= 0 or self.kpt_dim not in {2, 3}):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )

        # description recall fun
        def get_stats_desc():
            return (
                f"{self.num_found} images, {self.num_missing + self.num_empty} backgrounds, {self.num_corrupt} corrupt."
            )

        super().cache_labels(get_stats_desc)

        # update labels
        self.update_labels(include_class=self.classes)

        # set rect
        if self.rect:
            assert self.batch_size is not None
            self.set_rectangle()

    def update_labels(self, include_class: Optional[list]):
        """Update labels to include only these classes (optional)."""
        if include_class is None and not self.single_cls:
            return

        include_class_array = None
        if include_class is not None:
            include_class_array = np.array(include_class).reshape(1, -1)

        for i in range(len(self.labels)):
            if include_class_array is not None:
                cls = self.labels[i]["cls"]
                bboxes = self.labels[i]["bboxes"]
                segments = self.labels[i]["segments"]
                keypoints = self.labels[i]["keypoints"]

                j = (cls == include_class_array).any(1)
                self.labels[i]["cls"] = cls[j]
                self.labels[i]["bboxes"] = bboxes[j]
                if segments:
                    self.labels[i]["segments"] = [segments[si] for si, idx in enumerate(j) if idx]
                if keypoints is not None:
                    self.labels[i]["keypoints"] = keypoints[j]

            if self.single_cls:
                self.labels[i]["cls"][:, 0] = 0

    def label_read(self, index: int) -> tuple[np.ndarray | None]:
        """Read label from a specific path and verify the validity of relative data."""
        im_file, lb_file = self.img_files[index], self.label_files[index]
        # Number (missing, found, empty, corrupt), message, segments, keypoints
        segments, keypoints = [], None
        try:
            # Verify images
            im = Image.open(im_file)
            im.verify()  # PIL verify
            shape = (im.size[1], im.size[0])  # hw
            assert "." + im.format.lower() in IMG_FORMATS, f"Invalid image format {im.format}."

            # Verify labels
            if os.path.isfile(lb_file):
                self.num_found += 1  # label found
                with open(lb_file) as f:
                    lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                    if any(len(x) > 6 for x in lb) and (not self.use_keypoints):  # is segment
                        classes = np.array([x[0] for x in lb], dtype=np.float32)
                        segments = [
                            np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb
                        ]  # (cls, xy1, xy2, ...)
                        lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                    lb = np.array(lb, dtype=np.float32)
                if nl := len(lb):
                    if self.use_keypoints:
                        assert lb.shape[1] == (5 + self.nkpt * self.kpt_dim), (
                            f"Labels require {(5 + self.nkpt * self.kpt_dim)} columns each"
                        )
                        points = lb[:, 5:].reshape(-1, self.kpt_dim)[:, :2]
                    else:
                        assert lb.shape[1] == 5, f"Labels require 5 columns, {lb.shape[1]} columns detected"
                        points = lb[:, 1:]
                    assert points.max() <= 1, f"Non-normalized or out of bounds coordinates {points[points > 1]}"
                    assert lb.min() >= 0, f"Negative label values {lb[lb < 0]}"

                    # All labels
                    max_cls = lb[:, 0].max()  # max label count
                    assert max_cls <= self.nc, (
                        f"Label class {int(max_cls)} exceeds dataset class count {self.nc}. "
                        f"Possible class labels are 0-{self.nc - 1}"
                    )
                    _, i = np.unique(lb, axis=0, return_index=True)
                    if len(i) < nl:  # duplicate row check
                        lb = lb[i]  # remove duplicates
                        if segments:
                            segments = [segments[x] for x in i]
                        LOGGER.warning(f"{im_file}: {nl - len(i)} duplicate labels removed")
                else:
                    self.num_empty += 1  # label empty
                    lb = np.zeros((0, (5 + self.nkpt * self.kpt_dim) if self.use_keypoints else 5), dtype=np.float32)
            else:
                self.num_missing += 1  # label missing
                lb = np.zeros((0, (5 + self.nkpt * self.kpt_dim) if self.use_keypoints else 5), dtype=np.float32)
            if self.use_keypoints:
                keypoints = lb[:, 5:].reshape(-1, self.nkpt, self.kpt_dim)
                if self.kpt_dim == 2:
                    kpt_mask = np.where((keypoints[..., 0] < 0) | (keypoints[..., 1] < 0), 0.0, 1.0).astype(np.float32)
                    keypoints = np.concatenate([keypoints, kpt_mask[..., None]], axis=-1)  # (nl, nkpt, 3)
            lb = lb[:, :5]
            return im_file, lb, shape, segments, keypoints

        except Exception as e:
            self.num_corrupt += 1
            LOGGER.info(f"{im_file}: ignoring corrupt image/label: {e}")
            return None, None, None, None, None

    def label_format(self, label: tuple[np.ndarray | None]) -> dict[str, Any] | None:
        im_file, lb, shape, segments, keypoint = label
        if im_file:
            return {
                "im_file": im_file,
                "shape": shape,
                "cls": lb[:, 0:1],  # n, 1
                "bboxes": lb[:, 1:],  # n, 4
                "segments": segments,
                "keypoints": keypoint,
                "normalized": True,
                "bbox_format": "xywh",
            }
        else:
            return None

    def set_rectangle(self):
        """Sets the shape of bounding boxes for YOLO detections as rectangles."""
        bi = np.floor(np.arange(self.length) / self.batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches

        s = np.array([x.pop("shape") for x in self.labels])  # hw
        ar = s[:, 0] / s[:, 1]  # aspect ratio
        irect = ar.argsort()
        self.img_files = [self.img_files[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        ar = ar[irect]

        # Set training image shapes
        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride
        self.batch = bi  # batch index of image

    def cache_data(self):
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
                    self.imgs[i], self.im_hw0[i], self.im_hw[i] = x
                    if self.imgs[i]:
                        b += self.data[i].nbytes
                pbar.desc = f"Caching {self.mode} data ({b / gb:.5f}GB {storage})"
            pbar.close()

    def load_data(self, i: int, rect_mode: bool = True) -> tuple[np.ndarray | None]:
        """Loads 1 img and label from dataset index 'i', returns (img, label)."""
        im, imf, imfn = self.imgs[i], self.img_files[i], self.img_npy_files[i]

        # load data
        if im is None:  # cached in RAM
            if imfn.exists():  # cached in Disk
                try:
                    im = np.load(imfn)
                except Exception as e:
                    LOGGER.warning(f"Removing corrupt *.npy image file {imfn} due to: {e}")
                    Path(imfn).unlink(missing_ok=True)
                    im = self.file_read(imf)
            else:
                im = self.file_read(imf)

            h0, w0 = im.shape[:2]  # orig hw
            if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
                r = self.imgsz / max(h0, w0)  # ratio
                if r != 1:  # if sizes are not equal
                    w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):  # resize by stretching image to square imgsz
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

            # Add to buffer if training with augmentations
            if self.augment:
                self.imgs[i], self.im_hw0[i], self.im_hw[i] = (
                    im,
                    (h0, w0),
                    im.shape[:2],
                )  # im, hw_original, hw_resized
                self.mosaic_buffer.append(i)
                if 1 < len(self.mosaic_buffer) >= self.max_mosaic_buffer_length:  # prevent empty buffer
                    j = self.mosaic_buffer.pop(0)
                    if self.cache != "ram":
                        self.imgs[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            return im, (h0, w0), im.shape[:2]

        return im, self.im_hw0[i], self.im_hw[i]

    def get_sample(self, index: int) -> dict[str, Any]:
        """Get data and label information from the dataset."""
        sample = deepcopy(self.labels[index])
        sample.pop("shape", None)
        sample["img"], sample["ori_shape"], sample["resized_shape"] = self.load_data(index)
        sample["ratio_pad"] = (
            sample["resized_shape"][0] / sample["ori_shape"][0],
            sample["resized_shape"][1] / sample["ori_shape"][1],
        )  # for evaluation
        if self.rect:
            sample["rect_shape"] = self.batch_shapes[self.batch[index]]
        return self.update_annotations(sample)

    def update_annotations(self, sample: dict[str, Any]) -> dict[str, Any]:
        """
        Custom your annotations here.

        Note:
            cls is not with bboxes now, classification and semantic segmentation need an independent cls label
            Can also support classification and semantic segmentation by adding or removing dict keys there.
        """
        bboxes = sample.pop("bboxes")
        segments = sample.pop("segments", [])
        keypoints = sample.pop("keypoints", None)
        bbox_format = sample.pop("bbox_format")
        normalized = sample.pop("normalized")

        # NOTE: do NOT resample oriented boxes
        segment_resamples = 100 if self.use_obb else 1000
        if len(segments) > 0:
            # make sure segments interpolate correctly if original length is greater than segment_resamples
            max_len = max(len(s) for s in segments)
            segment_resamples = (max_len + 1) if segment_resamples < max_len else segment_resamples
            # list[np.array(segment_resamples, 2)] * num_samples
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        sample["instances"] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return sample

    def check_cache_ram(self, safety_margin=0.5):
        """Check data caching requirements vs available memory."""
        b, gb = 0, 1 << 30  # bytes of cached data, bytes per gigabytes
        n = min(self.length, 100)  # extrapolate from 100 random data
        skips = 0
        for _ in range(n):
            i = random.randint(0, self.length - 1)
            img = self.file_read(self.img_files[i])
            if img is None:
                skips += 1
                continue
            ratio = self.imgsz / max(img.shape[0], img.shape[1])
            b += img.nbytes * ratio**2
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

    def create_buffers(self, length):
        """Build buffers for data and labels storage."""
        super().create_buffers(length)

        self.im_hw0 = [None] * length
        self.register_buffer("im_hw0", self.im_hw0)
        self.im_hw = [None] * length
        self.register_buffer("im_hw", self.im_hw)

    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.mask_overlap,
                bgr=hyp.bgr if self.augment else 0.0,  # only affect training.
            )
        )
        return transforms

    def close_mosaic(self):
        """Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations."""
        hyp = deepcopy(self.hyp)
        hyp.mosaic = 0.0  # set mosaic ratio=0.0
        hyp.copy_paste = 0.0  # keep the same behavior as previous v8 close-mosaic
        hyp.mixup = 0.0  # keep the same behavior as previous v8 close-mosaic
        self.transforms = self.build_transforms(hyp)

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == "img":
                value = torch.stack(value, 0)
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)

        return new_batch


class YoloMultiModalDataset(MultiModalDatasetBase):
    """
    Dataset class for loading multispectral images with corresponding labels for object detection and/or segmentation
    tasks in YOLO format.

    Args:
        imgs (dict[str, np.ndarray | list[str]]): Multispectral image paths list or data arrays.
        labels (np.ndarray | list[str] | dict[str, list[str]]): Label path lists or label arrays.
        nc (int): Num of classes.
        imgsz (int): Image size. Defaults to 640.
        rect (bool): If True, rectangular training is used. Defaults to False.
        stride (int): Stride. Defaults to 32.
        pad (float): Padding. Defaults to 0.0.
        single_cls (bool): If True, single class training is used. Defaults to False.
        classes (list): List of included classes. Default is None.
        cache (bool): Cache data to RAM or disk during training. Defaults to False.
        augment (bool): If True, data augmentation is applied. Defaults to True.
        hyp (dict, optional): Hyperparameters to apply data augmentation. Defaults to None.
        batch_size (int): Size of batches. Defaults to None.
        fraction (float): Fraction of dataset to utilize. Default is 1.0 (use all data).
        modals (list[str]): List of modal names. Defaults to None.
        task ("detect", "pose", "segment"): The task of this dataset used for. Defaults to "detect".
        mode (Literal["train", "val", "test"]): The mode of the dataset.

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an detection or segmentation
        model utilizing multispectral images.
    """

    def __init__(
        self,
        imgs: dict[str, np.ndarray | list[str]],
        labels: np.ndarray | list[str] | dict[str, list[str] | np.ndarray],
        nc: int,
        imgsz: int = 640,
        rect: bool = False,
        stride: int = 32,
        pad: float = 0.5,
        single_cls: bool = False,
        classes: list[int] = None,
        cache: bool = False,
        augment: bool = True,
        hyp: dict[str, Any] | None = None,
        batch_size: int = 16,
        fraction: float = 1.0,
        modals: list[str] | None = None,
        dropout: bool = False,
        task: Literal["detect", "pose", "segment", "obb"] = "detect",
        mode: Literal["train", "val", "test"] = "train",
    ):
        """Initialize MultimodalDataset with given configuration and options."""
        self.imgsz = imgsz
        self.rect = rect
        self.stride = stride
        self.pad = pad
        self.single_cls = single_cls
        self.classes = classes
        self.nc = nc
        self.batch_size = batch_size

        self.task = task
        self.use_segments = self.task == "segment"
        self.use_keypoints = self.task == "pose"
        self.use_obb = self.task == "obb"

        super().__init__(
            data=imgs,
            labels=labels,
            cache=cache,
            hyp=hyp,
            fraction=fraction,
            augment=augment,
            modals=modals,
            dropout=dropout,
            mode=mode,
        )

    def setup_data_labels(self, data, labels):
        """Setup labels and data storage. Labels are always cached, data is cached conditionally."""
        # add mosaic_buffer used for images fusion
        self.mosaic_buffer = []
        self.max_mosaic_buffer_length = min((self.length, self.batch_size * 8, 1000)) if self.augment else 0

        super().setup_data_labels(data, labels)

    def cache_labels(self):
        """Cache labels from path list to buffers."""
        # statistical indicators
        self.num_missing = 0
        self.num_found = 0
        self.num_empty = 0
        self.num_corrupt = 0
        self.num_skip = 0

        # key points setting
        self.nkpt, self.kpt_dim = self.hyp.get("kpt_shape", (0, 0))
        if self.use_keypoints and (self.nkpt <= 0 or self.kpt_dim not in {2, 3}):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )

        def get_stats_desc():
            return (
                f"{self.num_found} couple of {tuple(self.modals)} images, "
                f"{self.num_missing + self.num_empty} backgrounds, "
                f"{self.num_skip} skip, "
                f"{self.num_corrupt} corrupt."
            )

        super().cache_labels(get_stats_desc)

        # update labels
        self.update_labels(include_class=self.classes)

        # set rect
        if self.rect:
            assert self.batch_size is not None
            self.set_rectangle()

    def label_read(self, i: int) -> tuple | None:
        """Read label"""
        # verify data
        data_accesses, shapes, files = self.verify_data(i)  # accessibilities, shape
        # Determine if should return None based on dropout mode
        if self.dropout:
            # In dropout mode: only return None if ALL data files are inaccessible
            corrupt = all(not access for access in data_accesses)
        else:
            # In normal mode: return None if ANY data file is inaccessible
            corrupt = any(not access for access in data_accesses)

        if corrupt:
            self.num_corrupt += 1
            LOGGER.info(f"Ignoring corrupt data index {i}.")
            return None

        # verify data shape and access
        if len(set(shapes)) != 1 or len(shapes) == 0:
            self.num_skip += 1
            LOGGER.info(f"Multispectral images mismatch in size, skip index {i}.")
            return None
        shape = shapes[0]

        # Read labels
        res = self.verify_label(i)
        if res is not None:
            return files, shape, *res
        else:
            return None

    def verify_data(self, i: int) -> tuple[list, list, dict]:
        """Verify data"""
        data_accesses, shapes, files = [], [], {}

        for name in self.data_names:
            file = self.data_files[name][i]
            files[f"{self.modal_mapping[name]}_file"] = file
            try:
                img = Image.open(file)
                img.verify()  # PIL verify
                if "." + img.format.lower() not in IMG_FORMATS:
                    raise TypeError(f"Invalid image format {img.format}.")
                data_accesses.append(True)
                shapes.append((img.size[1], img.size[0]))  # hw

            except Exception:
                data_accesses.append(False)

        return data_accesses, shapes, files

    def verify_label(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Verify labels"""

        # Labels with list format
        if isinstance(self.label_files, list):
            segments, keypoints = [], None
            lb_file = self.label_files[i]
            try:
                if os.path.isfile(lb_file):
                    self.num_found += 1  # label found
                    with open(lb_file) as f:
                        lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                        if any(len(x) > 6 for x in lb) and (not self.use_keypoints):  # is segment
                            classes = np.array([x[0] for x in lb], dtype=np.float32)
                            segments = [
                                np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb
                            ]  # (cls, xy1, xy2, ...)
                            lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                        lb = np.array(lb, dtype=np.float32)
                    if nl := len(lb):
                        if self.use_keypoints:
                            assert lb.shape[1] == (5 + self.nkpt * self.kpt_dim), (
                                f"Labels require {(5 + self.nkpt * self.kpt_dim)} columns each."
                            )
                            points = lb[:, 5:].reshape(-1, self.kpt_dim)[:, :2]
                        else:
                            assert lb.shape[1] == 5, f"Labels require 5 columns, {lb.shape[1]} columns detected."
                            points = lb[:, 1:]
                        assert points.max() <= 1, f"Non-normalized or out of bounds coordinates {points[points > 1]}."
                        assert lb.min() >= 0, f"Negative label values {lb[lb < 0]}."

                        # All labels
                        max_cls = lb[:, 0].max()  # max label count
                        assert max_cls <= self.nc, (
                            f"Label class {int(max_cls)} exceeds dataset class count {self.nc}."
                            f"Possible class labels are 0-{self.nc - 1}."
                        )
                        _, i = np.unique(lb, axis=0, return_index=True)
                        if len(i) < nl:  # duplicate row check
                            lb = lb[i]  # remove duplicates
                            if segments:
                                segments = [segments[x] for x in i]
                            LOGGER.warning(f"{lb_file}: {nl - len(i)} duplicate labels removed.")
                    else:
                        self.num_empty += 1  # label empty
                        lb = np.zeros(
                            (0, (5 + self.nkpt * self.kpt_dim) if self.use_keypoints else 5), dtype=np.float32
                        )
                else:
                    self.num_missing += 1  # label missing
                    lb = np.zeros((0, (5 + self.nkpt * self.kpt_dim) if self.use_keypoints else 5), dtype=np.float32)
                if self.use_keypoints:
                    keypoints = lb[:, 5:].reshape(-1, self.nkpt, self.kpt_dim)
                    if self.kpt_dim == 2:
                        kpt_mask = np.where((keypoints[..., 0] < 0) | (keypoints[..., 1] < 0), 0.0, 1.0).astype(
                            np.float32
                        )
                        keypoints = np.concatenate([keypoints, kpt_mask[..., None]], axis=-1)  # (nl, nkpt, 3)
                lb = lb[:, :5]
                return lb, segments, keypoints

            except Exception as e:
                self.num_corrupt += 1
                LOGGER.info(f"Ignoring corrupt label at {lb_file}: {e}.")
                return None

        # Labels with dict format
        if isinstance(self.label_files, dict):
            # statistical indicators
            lb_found, lb_missing, lb_empty, lb_corrupt = [], [], [], []
            lb_ls, segments_ls, keypoints_ls = [], [], []

            for name in self.label_files:
                segs, kpts = [], None
                lb_file = self.label_files[name][i]
                try:
                    if os.path.isfile(lb_file):
                        lb_found.append(True)  # label found
                        with open(lb_file) as f:
                            lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                            if any(len(x) > 6 for x in lb) and (not self.use_keypoints):  # is segment
                                classes = np.array([x[0] for x in lb], dtype=np.float32)
                                segs = [
                                    np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb
                                ]  # (cls, xy1, xy2, ...)
                                lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segs)), 1)  # (cls, xywh)
                            lb = np.array(lb, dtype=np.float32)
                        if nl := len(lb):
                            if self.use_keypoints:
                                assert lb.shape[1] == (5 + self.nkpt * self.kpt_dim), (
                                    f"{lb_file}: Labels require {(5 + self.nkpt * self.kpt_dim)} columns each."
                                )
                                points = lb[:, 5:].reshape(-1, self.kpt_dim)[:, :2]
                            else:
                                assert lb.shape[1] == 5, (
                                    f"{lb_file}: Labels require 5 columns, {lb.shape[1]} columns detected."
                                )
                                points = lb[:, 1:]
                            assert points.max() <= 1, (
                                f"{lb_file}: Non-normalized or out of bounds coordinates {points[points > 1]}."
                            )
                            assert lb.min() >= 0, f"{lb_file}: Negative label values {lb[lb < 0]}."

                            # All labels
                            max_cls = lb[:, 0].max()  # max label count
                            assert max_cls <= self.nc, (
                                f"{lb_file}: Label class {int(max_cls)} exceeds dataset class count {self.nc}."
                                f"Possible class labels are 0-{self.nc - 1}."
                            )
                            _, i = np.unique(lb, axis=0, return_index=True)
                            if len(i) < nl:  # duplicate row check
                                lb = lb[i]  # remove duplicates
                                if len(segs) > 0:
                                    segs = [segs[x] for x in i]
                                LOGGER.warning(f"{lb_file}: {nl - len(i)} duplicate labels removed.")
                        else:
                            lb_empty.append(True)  # label empty
                            lb = np.zeros(
                                (0, (5 + self.nkpt * self.kpt_dim) if self.use_keypoints else 5), dtype=np.float32
                            )
                    else:
                        lb_missing.append(True)  # label missing
                        lb = np.zeros(
                            (0, (5 + self.nkpt * self.kpt_dim) if self.use_keypoints else 5), dtype=np.float32
                        )
                    if self.use_keypoints:
                        kpts = lb[:, 5:].reshape(-1, self.nkpt, self.kpt_dim)
                        if self.kpt_dim == 2:
                            kpt_mask = np.where((kpts[..., 0] < 0) | (kpts[..., 1] < 0), 0.0, 1.0).astype(np.float32)
                            kpts = np.concatenate([kpts, kpt_mask[..., None]], axis=-1)  # (nl, nkpt, 3)

                    lb = lb[:, :5]
                    lb_ls.append(lb), segments_ls.append(segs)
                    if kpts is not None:
                        keypoints_ls.append(kpts)

                except Exception:
                    LOGGER.warning(f"{lb_file}: Ignoring corrupted label in multi-labels.")
                    lb_corrupt.append(True)

            if self.dropout:
                if all(lb_corrupt):
                    self.num_corrupt += 1
                    LOGGER.info(f"Ignoring corrupt multi-labels with index '{i}'.")
                    return None
                else:
                    self.num_found += 1 if any(lb_found) else 0
                    self.num_empty += 1 if all(lb_empty) else 0
                    self.num_missing += 1 if all(lb_missing) else 0

                    lb = np.concatenate(lb_ls, axis=0)
                    segments = list(chain(*segments_ls))
                    keypoints = np.concatenate(keypoints_ls, axis=0) if len(keypoints_ls) > 0 else None

                    lb_unique, unique_idx = np.unique(lb, axis=0, return_index=True)
                    if len(unique_idx) < len(lb):  # duplicate row check
                        lb = lb_unique  # remove duplicates
                        if len(segments) > 0:
                            segments = [segments[j] for j in unique_idx]
                        if keypoints is not None and len(keypoints) > 0:
                            keypoints = keypoints[unique_idx]
                        LOGGER.warning("Duplicate labels removed in multi-labels.")

                    return lb, segments, keypoints

            else:
                if any(lb_corrupt):
                    self.num_corrupt += 1
                    LOGGER.info(f"Ignoring corrupt multi-labels with index '{i}'.")
                    return None
                else:
                    self.num_found += 1 if all(lb_found) else 0
                    self.num_empty += 1 if all(lb_empty) else 0
                    self.num_missing += 1 if all(lb_missing) else 0

                    lb = np.concatenate(lb_ls, axis=0)
                    segments = list(chain(*segments_ls))
                    keypoints = np.concatenate(keypoints_ls, axis=0) if len(keypoints_ls) > 0 else None

                    lb_unique, unique_idx = np.unique(lb, axis=0, return_index=True)
                    if len(unique_idx) < len(lb):  # duplicate row check
                        lb = lb_unique  # remove duplicates
                        if len(segments) > 0:
                            segments = [segments[j] for j in unique_idx]
                        if keypoints is not None and len(keypoints) > 0:
                            keypoints = keypoints[unique_idx]
                        LOGGER.warning("Duplicate labels removed in multi-labels.")

                    return lb, segments, keypoints

    def label_format(self, label: tuple | None) -> dict[str, Any] | None:
        """Custom multimodal yolo label format.

        Example:
            {
                # data file paths
                "< modal_1 >_file": "/home/yangxf/...",    # modal_n is the modal name in self.modals
                "< modal_2 >_file": "/home/yangxf/...",
                ...
                "shape": (640, 512),
                "cls": np.array([[0], [1], [25], ...]),
                "bboxes": ...,
                "segments": ...,
                "keypoints": ...,
                "normalized": True,
                "bbox_format": "xywh",
            }
        """
        if label is None:
            return None

        file_dict, shape, lb, segments, keypoint = label
        file_dict.update(
            {
                "shape": shape,
                "cls": lb[:, 0:1],  # n, 1
                "bboxes": lb[:, 1:],  # n, 4
                "segments": segments,
                "keypoints": keypoint,
                "normalized": True,
                "bbox_format": "xywh",
            }
        )
        return file_dict

    def update_labels(self, include_class: Optional[list]) -> None:
        """Update labels to include only these classes (optional)."""
        if include_class is None and not self.single_cls:
            return

        include_class_array = None
        if include_class is not None:
            include_class_array = np.array(include_class).reshape(1, -1)

        for i in range(len(self.labels)):
            if include_class_array is not None:
                cls = self.labels[i]["cls"]
                bboxes = self.labels[i]["bboxes"]
                segments = self.labels[i]["segments"]
                keypoints = self.labels[i]["keypoints"]

                j = (cls == include_class_array).any(1)
                self.labels[i]["cls"] = cls[j]
                self.labels[i]["bboxes"] = bboxes[j]
                if segments:
                    self.labels[i]["segments"] = [segments[si] for si, idx in enumerate(j) if idx]
                if keypoints is not None:
                    self.labels[i]["keypoints"] = keypoints[j]

            if self.single_cls:
                self.labels[i]["cls"][:, 0] = 0

    def set_rectangle(self):
        """Sets the shape of bounding boxes for YOLO detections as rectangles."""
        bi = np.floor(np.arange(self.length) / self.batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches

        s = np.array([x.pop("shape") for x in self.labels])  # hw
        ar = s[:, 0] / s[:, 1]  # aspect ratio
        irect = ar.argsort()

        # Rearrange the data files
        for fs in self.data_files.values():
            fs[:] = [fs[i] for i in irect]

        # Rearrange the labels
        if isinstance(self.labels, list):
            self.labels[:] = [self.labels[i] for i in irect]
        elif isinstance(self.labels, dict):
            for lbs in self.labels.values():
                lbs[:] = [lbs[i] for i in irect]

        ar = ar[irect]

        # Set training image shapes
        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride
        self.batch = bi  # batch index of image

    def cache_data(self):
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
                            self.data[name][i] = x[0][self.modal_mapping[name]]
                            self.hw0[i] = x[1]
                            self.hw[i] = x[2]
                            b += self.data[name][i].nbytes
                pbar.desc = f"Caching {self.mode} data ({b / gb:.5f}GB {storage})"
            pbar.close()

    def load_data(self, i: int, rect_mode: bool = True) -> tuple[dict, tuple, tuple]:
        """Loads 1 img and label from dataset index 'i', returns (img, label)."""
        data, hw0, hw = {}, None, None

        # Whether data has been loaded into ram
        cached_ram = next(iter(self.data.values()))[i] is not None

        if cached_ram:  # cached in ram
            for name in self.data_names:
                data[self.modal_mapping[name]] = self.data[name][i]
            hw = self.hw[i]
            hw0 = self.hw0[i]

        else:  # not cached in ram
            for name in self.data_names:
                dtf, dtfn = self.data_files[name][i], self.data_npy_files[name][i]
                # read data
                if dtfn.exists():  # cached in Disk
                    try:
                        dt = np.load(dtfn)
                    except Exception as e:
                        LOGGER.warning(f"Removing corrupt *.npy image file {dtfn} due to: {e}")
                        Path(dtfn).unlink(missing_ok=True)
                        dt = DatasetBase.file_read(dtf)
                else:
                    dt = DatasetBase.file_read(dtf)

                data[self.modal_mapping[name]] = dt  # may be None, dropout mode

            shapes = set([dt.shape[:2] for dt in data.values()])
            if len(shapes) != 1:
                raise ValueError(
                    "The hight and width of the multispectral images actually loaded from the disk are inconsistent."
                )

            h0, w0 = next(iter(shapes))
            hw0 = (h0, w0)
            # deal with load data
            if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
                r = self.imgsz / max(h0, w0)  # ratio
                if r != 1:  # if sizes are not equal
                    w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                else:
                    w, h = w0, h0
            else:
                if not (h0 == w0 == self.imgsz):
                    w = h = self.imgsz
                else:
                    w, h = w0, h0
            hw = (h, w)

            for modal, dt in data.items():
                if dt is not None:
                    data[modal] = cv2.resize(dt, (w, h), interpolation=cv2.INTER_LINEAR)
                else:
                    data[modal] = np.zeros((w, h))

            if self.augment:
                for name in self.data_names:
                    self.data[name][i] = data[self.modal_mapping[name]]
                self.hw0[i] = hw0
                self.hw[i] = hw
                self.mosaic_buffer.append(i)
                if 1 < len(self.mosaic_buffer) >= self.max_mosaic_buffer_length:  # prevent empty buffer
                    j = self.mosaic_buffer.pop(0)
                    if self.cache != "ram":
                        for name in self.data_names:
                            self.data[name][j] = None
                        self.hw0[j], self.hw[j] = None, None

        return data, hw0, hw

    def __getitem__(self, index) -> dict[str, Any]:
        sample = self.get_sample(index)
        if self.transforms:
            sample = self.transforms(sample)  # The transform must take label as input.
        return sample

    def get_sample(self, index: int) -> dict[str, Any]:
        """Get data and label information from the dataset."""
        sample = deepcopy(self.labels[index])
        sample.pop("shape", None)
        data, sample["ori_shape"], sample["resized_shape"] = self.load_data(index)
        sample["ratio_pad"] = (
            sample["resized_shape"][0] / sample["ori_shape"][0],
            sample["resized_shape"][1] / sample["ori_shape"][1],
        )  # for evaluation
        if self.rect:
            sample["rect_shape"] = self.batch_shapes[self.batch[index]]

        sample.update(data)

        return self.update_annotations(sample)

    def update_annotations(self, sample: dict[str, Any]) -> dict[str, Any]:
        """
        Custom your label format here.

        Note:
            cls is not with bboxes now, classification and semantic segmentation need an independent cls label
            Can also support classification and semantic segmentation by adding or removing dict keys there.
        """
        bboxes = sample.pop("bboxes")
        segments = sample.pop("segments", [])
        keypoints = sample.pop("keypoints", None)
        bbox_format = sample.pop("bbox_format")
        normalized = sample.pop("normalized")

        # NOTE: do NOT resample oriented boxes
        segment_resamples = 100 if self.use_obb else 1000
        if len(segments) > 0:
            # make sure segments interpolate correctly if original length is greater than segment_resamples
            max_len = max(len(s) for s in segments)
            segment_resamples = (max_len + 1) if segment_resamples < max_len else segment_resamples
            # list[np.array(segment_resamples, 2)] * num_samples
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        sample["instances"] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return sample

    def check_cache_ram(self, safety_margin=0.5):
        """Check data caching requirements vs available memory."""
        b, gb = 0, 1 << 30  # bytes of cached data, bytes per gigabytes
        n = min(self.length, 100)  # extrapolate from 100 random data
        skips = 0
        for _ in range(n):
            i = random.randint(0, self.length - 1)
            data = []
            for name in self.data_names:
                dt = DatasetBase.file_read(self.data_files[name][i])
                if dt is None:
                    data = []
                    skips += 1
                    break
                else:
                    data.append(dt)
            b += np.sum([d.nbytes * (self.imgsz / max(d.shape[0], d.shape[1])) ** 2 for d in data])
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

    def create_buffers(
        self,
        length: int,
        labels: np.ndarray | list[str] | dict[str, list[str] | np.ndarray],
    ) -> None:
        """Build buffers for data and labels storage."""
        super().create_buffers(length, labels)

        self.hw0 = [None] * length
        self.register_buffer("hw0", self.hw0)
        self.hw = [None] * length
        self.register_buffer("hw", self.hw)

    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        ...

    def close_mosaic(self):
        """Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations."""
        hyp = deepcopy(self.hyp)
        hyp.mosaic = 0.0  # set mosaic ratio=0.0
        hyp.copy_paste = 0.0  # keep the same behavior as previous v8 close-mosaic
        hyp.mixup = 0.0  # keep the same behavior as previous v8 close-mosaic
        self.transforms = self.build_transforms(hyp)

    def collate_fn(self, batch):
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k in self.modals:
                value = torch.stack(value, 0)
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch
