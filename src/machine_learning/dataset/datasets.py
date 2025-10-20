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
from multiprocessing.pool import ThreadPool

from machine_learning.utils.logger import LOGGER
from machine_learning.types.aliases import FilePath
from machine_learning.dataset.base import DatasetBase
from machine_learning.utils.constants import IMG_FORMATS, NUM_THREADS
from machine_learning.utils.augment import Compose, Format, Instances, LetterBox, v8_transforms

from ultralytics.utils.ops import segments2boxes
from ultralytics.utils.ops import resample_segments


class YoloDataset(DatasetBase):
    """
    YoloDataset class for loading object detection and/or segmentation labels in YOLO format.

    Args:
        imgs (list[str]): Path list to imgs or imgs itself with np.ndarray format.
        labels (list[str]): Path list to the labels or labels itself with np.ndarray format.
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
        mode (Literal["train", "val", "test"]): The mode of the dataset.

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
    """

    def __init__(
        self,
        imgs: np.ndarray | list[FilePath],
        labels: np.ndarray | list[FilePath],
        imgsz: int = 640,
        num_cls: int = 80,
        task: Literal["detect", "pose", "segment"] = "detect",
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
        mode: Literal["train", "val", "test"] = "train",
    ):
        """Initialize BaseDataset with given configuration and options."""
        self.imgsz = imgsz
        self.rect = rect
        self.stride = stride
        self.pad = pad
        self.single_cls = single_cls
        self.classes = classes
        self.num_cls = num_cls

        self.task = task
        self.use_segments = self.task == "segment"
        self.use_keypoints = self.task == "pose"
        self.use_obb = self.task == "obb"

        super().__init__(
            data=imgs,
            labels=labels,
            cache=cache,
            augment=augment,
            hyp=hyp,
            batch_size=batch_size,
            fraction=fraction,
            mode=mode,
        )  # cache imgs and labels
        self.update_labels(include_class=classes)

        # Buffer thread for data fusion
        self.mosaic_buffer = []
        self.max_mosaic_buffer_length = min((self.length, self.batch_size * 8, 1000)) if self.augment else 0

        if self.rect:
            assert self.batch_size is not None
            self.set_rectangle()

    @property
    def imgs(self) -> list:
        return self.data

    @property
    def img_files(self) -> list:
        return self.data_files

    @property
    def img_npy_files(self) -> list:
        return self.data_npy_files

    def get_labels(self):
        """Get labels from path list to buffers."""
        # statistical indicators
        self.nm = 0
        self.nf = 0
        self.ne = 0
        self.nc = 0

        # key points setting
        self.nkpt, self.ndim = self.hyp.get("kpt_shape", (0, 0))
        if self.use_keypoints and (self.nkpt <= 0 or self.ndim not in {2, 3}):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )

        def get_stats_desc():
            return f"{self.nf} images, {self.nm + self.ne} backgrounds, {self.nc} corrupt"

        super().get_labels(get_stats_desc)

    def lread(self, index: int) -> tuple[np.ndarray | None]:
        """Read label"""
        im_file, lb_file = self.data_files[index], self.label_files[index]
        # Number (missing, found, empty, corrupt), message, segments, keypoints
        segments, keypoints = [], None
        try:
            # Verify images
            im = Image.open(im_file)
            im.verify()  # PIL verify
            shape = (im.size[1], im.size[0])  # hw
            assert "." + im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}."

            # Verify labels
            if os.path.isfile(lb_file):
                self.nf += 1  # label found
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
                        assert lb.shape[1] == (5 + self.nkpt * self.ndim), (
                            f"labels require {(5 + self.nkpt * self.ndim)} columns each"
                        )
                        points = lb[:, 5:].reshape(-1, self.ndim)[:, :2]
                    else:
                        assert lb.shape[1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"
                        points = lb[:, 1:]
                    assert points.max() <= 1, f"non-normalized or out of bounds coordinates {points[points > 1]}"
                    assert lb.min() >= 0, f"negative label values {lb[lb < 0]}"

                    # All labels
                    max_cls = lb[:, 0].max()  # max label count
                    assert max_cls <= self.num_cls, (
                        f"Label class {int(max_cls)} exceeds dataset class count {self.num_cls}. "
                        f"Possible class labels are 0-{self.num_cls - 1}"
                    )
                    _, i = np.unique(lb, axis=0, return_index=True)
                    if len(i) < nl:  # duplicate row check
                        lb = lb[i]  # remove duplicates
                        if segments:
                            segments = [segments[x] for x in i]
                        LOGGER.warning(f"{im_file}: {nl - len(i)} duplicate labels removed")
                else:
                    self.ne += 1  # label empty
                    lb = np.zeros((0, (5 + self.nkpt * self.ndim) if self.use_keypoints else 5), dtype=np.float32)
            else:
                self.nm += 1  # label missing
                lb = np.zeros((0, (5 + self.nkpt * self.ndim) if self.use_keypoints else 5), dtype=np.float32)
            if self.use_keypoints:
                keypoints = lb[:, 5:].reshape(-1, self.nkpt, self.ndim)
                if self.ndim == 2:
                    kpt_mask = np.where((keypoints[..., 0] < 0) | (keypoints[..., 1] < 0), 0.0, 1.0).astype(np.float32)
                    keypoints = np.concatenate([keypoints, kpt_mask[..., None]], axis=-1)  # (nl, nkpt, 3)
            lb = lb[:, :5]
            return im_file, lb, shape, segments, keypoints

        except Exception as e:
            self.nc += 1
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

    def update_labels(self, include_class: Optional[list]):
        """Update labels to include only these classes (optional)."""
        include_class_array = np.array(include_class).reshape(1, -1)
        for i in range(len(self.labels)):
            if include_class is not None:
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
        self.im_files = [self.im_files[i] for i in irect]
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

            if im is not None:
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

            else:
                self.corrupt_idx.add(i)  # data corrupt

                return None, None, None

        return im, self.im_hw0[i], self.im_hw[i]

    def get_data_and_label(self, index: int) -> dict[str, Any]:
        return self.get_image_and_label(index)

    def get_image_and_label(self, index: int) -> dict[str, Any]:
        """Get and return label information from the dataset."""
        label = deepcopy(self.labels[index])
        label.pop("shape", None)
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_data(index)
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # for evaluation
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label)

    def update_labels_info(self, label: dict[str, Any]) -> dict[str, Any]:
        """
        Custom your label format here.

        Note:
            cls is not with bboxes now, classification and semantic segmentation need an independent cls label
            Can also support classification and semantic segmentation by adding or removing dict keys there.
        """
        bboxes = label.pop("bboxes")
        segments = label.pop("segments", [])
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")

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
        label["instances"] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

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


class MultimodalDataset(DatasetBase):
    def __init__(
        self,
        data: dict[str, Union[np.ndarray, list[str]]],
        labels: Union[np.ndarray, list[str], dict[str, Union[np.ndarray, list[str]]]],
        cache: bool = False,
        augment: bool = True,
        hyp: dict[str, Any] = {},
        batch_size: int = 16,
        fraction: float = 1.0,
        mode: Literal["train", "val", "test"] = "train",
    ):
        """Initialize MultimodalDataset with given configuration and options."""
        self._modal_names = data.keys()
        self._label_names = labels.keys() if isinstance(labels, dict) else None

        super().__init__(
            data=data,
            labels=labels,
            cache=cache,
            augment=augment,
            hyp=hyp,
            batch_size=batch_size,
            fraction=fraction,
            mode=mode,
        )

    @property
    def modal_names(self) -> list[str]:
        return self._modal_names

    @property
    def label_names(self) -> list[str] | None:
        return self._label_names

    def create_buffers(self, length: int) -> None:
        """Build buffers for data and labels storage."""
        self.buffers = {}

        # data
        for modal in self.modal_names:
            setattr(self, f"{modal}", [None] * length)
            setattr(self, f"{modal}_files", [None] * length)
            setattr(self, f"{modal}_npy_files", [None] * length)
            self.buffers[f"{modal}"] = getattr(self, f"{modal}")
            self.buffers[f"{modal}_files"] = getattr(self, f"{modal}_files")
            self.buffers[f"{modal}_npy_files"] = getattr(self, f"{modal}_npy_files")

        # labels
        if self.label_names:
            for label in self.label_names:
                setattr(self, f"{label}_label", [None] * length)
                setattr(self, f"{label}_label_files", [None] * length)
                self.buffers[f"{label}_label"] = getattr(self, f"{label}_label")
                self.buffers[f"{label}_label_files"] = getattr(self, f"{label}_label_files")
        else:
            self.labels = [None] * length
            self.label_files = [None] * length

            self.buffers["labels"] = self.labels
            self.buffers["label_files"] = self.label_files

    def init_cache(
        self,
        data: dict[str, Union[list[str], np.ndarray]],
        labels: Union[np.ndarray, list[str], dict[str, Union[np.ndarray, list[str]]]],
    ) -> None:
        # np.ndarray
        if isinstance(data.values()[0], np.ndarray) and (
            isinstance(labels.values()[0], np.ndarray) if isinstance(labels, dict) else isinstance(labels, np.ndarray)
        ):
            self.get_labels_np(labels)
            self.cache_data_np(data)

        # list[str]
        elif isinstance(data.values()[0], list) and (
            isinstance(labels.values()[0], list) if isinstance(labels, dict) else isinstance(labels, list)
        ):
            self.data_files = {modal: data[modal] for modal in self.modal_names}
            self.label_files = {label: labels[label] for label in self.label_names}

            # labels
            self.get_labels()

            # data
            self.data_npy_files = {
                modal: [Path(f).with_suffix(".npy") for f in data[modal]] for modal in self.modal_names
            }

            if self.cache == "ram" and self.check_cache_ram():
                self.cache_data()
            elif self.cache == "disk" and self.check_cache_disk():
                self.cache_data()

        else:
            raise TypeError(
                "The input data and labels must be the same of type (dict[str, np.ndarray] or dict[str, list])."
            )

    def get_labels_np(self, labels: dict[str, np.ndarray]) -> None:
        """Get labels from matrix input to buffers."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        LOGGER.info(f"Caching {self.mode} labels from matrix input...")
        pbar = tqdm(range(self.length))
        for i in pbar:
            for label in self.label_names:
                label = self.label_format(labels[label])
                getattr(self, f"label_{label}")[i] = label
                b += asizeof.asizeof(label)
            pbar.desc = f"Caching {self.mode} labels ({b / gb:.5f}GB)"
        pbar.close()

    def get_labels(self, desc_func: Callable) -> None:
        """Get labels from path list to buffers."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        LOGGER.info(f"Caching {self.mode} labels...")
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(self.cache_labels, range(self.length))
            pbar = tqdm(enumerate(results), total=self.length)
            for _, lb in pbar:
                if lb:
                    b += asizeof.asizeof(lb)
                pbar.desc = f"Caching {self.mode} labels ({b / gb:.5f}GB) " + desc_func()
            pbar.close()

    def cache_labels(self, i: int) -> dict[str, Any] | None:
        """Cache label from paths to ram for faster loading."""
        labels = self.lread(i)
        labels = self.label_format(labels)
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
                    dt = np.load(dtfn)
                except Exception as e:
                    LOGGER.warning(f"Removing corrupt *.npy image file {dtfn} due to: {e}")
                    Path(dtfn).unlink(missing_ok=True)
                    dt = self.file_read(dtf)
            else:
                dt = self.file_read(dtf)

            if dt is None:  # data corrupt
                self.corrupt_idx.add(i)

            return dt

        return dt

    def __len__(self):
        """Returns the length of the labels list for the dataset."""
        return self.length

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns transformed label information for given index."""
        data, label = self.get_data_and_label(index)
        if self.transforms:
            data = self.transforms(data)  # The transform must take label as input.
        return data, label

    def get_data_and_label(self, index: int) -> tuple[np.ndarray, np.ndarray]:
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

    def lread(self, index: int) -> dict[str, np.ndarray | None]:
        """Users implement their own label reading logic."""
        labels = {}
        for lb in self.label_names:
            label_path = getattr(self, f"{lb}_label_files")[index]
            label = self.file_read(label_path)
            labels[lb] = label

        return labels

    @staticmethod
    def file_read(path: FilePath) -> np.ndarray | None:
        path = Path(path)
        extension = path.suffix.lower()

        if not path.exists():
            LOGGER.error(f"File does not exist: {path}")
            return None

        try:
            if extension in IMG_FORMATS:  # imgs
                data = cv2.imread(str(path))  # bgr
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

    def register_buffer(self, name: str, buffer: list) -> None:
        length = len(getattr(self, f"{self.modal_names[0]}_data"))

        if len(buffer) != length:
            raise ValueError(f"The length of {name} buffer must be equal to data buffer.")

        if name not in self.buffers.keys():
            self.buffers[name] = buffer
        else:
            raise KeyError(f"Buffer {name} already exists.")

    def update_buffers(self) -> None:
        """Update the buffer and delete invalid data (None)."""
        if self.corrupt_idx:
            LOGGER.info(f"Removing invalid items from buffers...: {[id for id in self.corrupt_idx]}")

            for i in sorted(self.corrupt_idx, reverse=True):
                self.remove_item(i)
            self.corrupt_idx.clear()

        self.length = len(getattr(self, f"{self.label_names[0]}_label"))

    def remove_item(self, i: int) -> None:
        """Remove a item of buffers according to the index."""
        for buffer in self.buffers.values():
            buffer.pop(i)
