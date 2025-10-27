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
from machine_learning.dataset.base import DatasetBase, MMDatasetBase
from machine_learning.utils.constants import IMG_FORMATS, NUM_THREADS
from machine_learning.utils.transforms import Compose, Format, Instances, LetterBox, v8_transforms

from ultralytics.utils.ops import segments2boxes
from ultralytics.utils.ops import resample_segments


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
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
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
        task: Literal["detect", "pose", "segment"] = "detect",
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
            augment=augment,
            hyp=hyp,
            fraction=fraction,
            mode=mode,
        )  # cache imgs and labels

        # Buffer thread for data fusion
        self.mosaic_buffer = []
        self.max_mosaic_buffer_length = min((self.length, self.batch_size * 8, 1000)) if self.augment else 0

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

    def cache_labels(self):
        """Get labels from path list to buffers."""
        # statistical indicators
        self.num_missing = 0
        self.num_found = 0
        self.num_empty = 0
        self.num_corrupt = 0

        # key points setting
        self.nkpt, self.ndim = self.hyp.get("kpt_shape", (0, 0))
        if self.use_keypoints and (self.nkpt <= 0 or self.ndim not in {2, 3}):
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
                        assert lb.shape[1] == (5 + self.nkpt * self.ndim), (
                            f"Labels require {(5 + self.nkpt * self.ndim)} columns each"
                        )
                        points = lb[:, 5:].reshape(-1, self.ndim)[:, :2]
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
                    lb = np.zeros((0, (5 + self.nkpt * self.ndim) if self.use_keypoints else 5), dtype=np.float32)
            else:
                self.num_missing += 1  # label missing
                lb = np.zeros((0, (5 + self.nkpt * self.ndim) if self.use_keypoints else 5), dtype=np.float32)
            if self.use_keypoints:
                keypoints = lb[:, 5:].reshape(-1, self.nkpt, self.ndim)
                if self.ndim == 2:
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
        return self.update_labels_info(sample)

    def update_labels_info(self, sample: dict[str, Any]) -> dict[str, Any]:
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


class ImgIrDataset(MMDatasetBase):
    """
    Dataset class for loading RGB and IR images with corresponding labels for object detection and/or segmentation tasks in YOLO format.
    """

    def __init__(
        self,
        data: dict[str, np.ndarray | list[str]],
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
        modal_names: list[str] | None = None,
        task: Literal["detect", "pose", "segment"] = "detect",
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
            data=data,
            labels=labels,
            cache=cache,
            augment=augment,
            hyp=hyp,
            batch_size=batch_size,
            fraction=fraction,
            modal_names=modal_names,
            mode=mode,
        )

        # Buffer thread for data fusion
        self.augment_buffer = []
        self.max_augment_buffer_length = min((self.length, self.batch_size * 8, 1000)) if self.augment else 0

    @property
    def imgs(self) -> list:
        return self.data["img"]

    @property
    def img_files(self) -> list:
        return self.data_files["img"]

    @property
    def img_npy_files(self) -> list:
        return self.data_npy_files["img"]

    @property
    def irs(self) -> list:
        return self.data["ir"]

    @property
    def ir_files(self) -> list:
        return self.data_files["ir"]

    @property
    def ir_npy_files(self) -> list:
        return self.data_npy_files["ir"]

    def get_labels(self):
        """Get labels from path list to buffers."""
        # statistical indicators
        self.num_missing = 0
        self.num_found = 0
        self.num_empty = 0
        self.num_corrupt = 0

        # key points setting
        self.nkpt, self.ndim = self.hyp.get("kpt_shape", (0, 0))
        if self.use_keypoints and (self.nkpt <= 0 or self.ndim not in {2, 3}):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )

        def get_stats_desc():
            return (
                f"{self.num_found} images, {self.num_missing + self.num_empty} backgrounds, {self.num_corrupt} corrupt."
            )

        super().get_labels(get_stats_desc)

        # update labels
        self.update_labels(include_class=self.classes)

        # set rect
        if self.rect:
            assert self.batch_size is not None
            self.set_rectangle()

    def lread(self, index: int) -> tuple:
        """Read label"""
        if isinstance(self.label_files, list):
            im_file, ir_file, lb_file = self.img_files[index], self.ir_files[index], self.label_files[index]
            # Number (missing, found, empty, corrupt), segments, keypoints
            segments, keypoints = [], None
            try:
                # Verify images
                im = Image.open(im_file)
                im.verify()  # PIL verify

                # Verify irs
                ir = Image.open(ir_file)
                ir.verify()  # PIL verify

                assert im.size == ir.size, "Image and IR size mismatch."
                shape = (im.size[1], im.size[0])  # hw
                assert "." + im.format.lower() in IMG_FORMATS, f"Invalid image format {im.format}."
                assert "." + ir.format.lower() in IMG_FORMATS, f"Invalid ir format {ir.format}."

                # Verify labels
                lb, segments, keypoints = self.verify_label(lb_file)
                return im_file, ir_file, lb, shape, segments, keypoints

            except Exception as e:
                self.num_corrupt += 1
                LOGGER.info(f"{im_file}/{ir_file}: ignoring corrupt image/ir/label: {e}.")
                return None, None, None, None, None, None

        elif isinstance(self.label_files, dict):
            im_file, ir_file, im_lb_file, ir_lb_file = (
                self.img_files[index],
                self.ir_files[index],
                self.label_files["img"][index],
                self.label_files["ir"][index],
            )
            # Number (missing, found, empty, corrupt), segments, keypoints
            segments, keypoints = [], None

            try:
                # Verify images
                im = Image.open(im_file)
                im.verify()  # PIL verify

                # Verify irs
                ir = Image.open(ir_file)
                ir.verify()  # PIL verify

                assert im.size == ir.size, "Image and IR size mismatch."
                shape = (im.size[1], im.size[0])  # hw
                assert "." + im.format.lower() in IMG_FORMATS, f"Invalid image format {im.format}."
                assert "." + ir.format.lower() in IMG_FORMATS, f"Invalid ir format {ir.format}."

                # Verify labels
                im_lb, im_segments, im_keypoints = self.verify_label(im_lb_file)
                ir_lb, ir_segments, ir_keypoints = self.verify_label(ir_lb_file)
                lb = np.concatenate((im_lb, ir_lb), axis=0)
                segments = np.concatenate((im_segments, ir_segments), axis=0)
                keypoints = np.concatenate((im_keypoints, ir_keypoints), axis=0)

                return im_file, ir_file, lb, shape, segments, keypoints

            except Exception as e:
                self.num_corrupt += 1
                LOGGER.info(f"{im_file}/{ir_file}: ignoring corrupt image/ir/label: {e}.")
                return None, None, None, None, None, None

    def verify_label(self, lb_file: str) -> tuple:
        # Verify labels
        if os.path.isfile(lb_file):
            self.num_found += 1  # label found
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in lb) and (not self.use_keypoints):  # is segment
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1, xy2, ...)
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                lb = np.array(lb, dtype=np.float32)
            if nl := len(lb):
                if self.use_keypoints:
                    assert lb.shape[1] == (5 + self.nkpt * self.ndim), (
                        f"Labels require {(5 + self.nkpt * self.ndim)} columns each."
                    )
                    points = lb[:, 5:].reshape(-1, self.ndim)[:, :2]
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
                lb = np.zeros((0, (5 + self.nkpt * self.ndim) if self.use_keypoints else 5), dtype=np.float32)
        else:
            self.num_missing += 1  # label missing
            lb = np.zeros((0, (5 + self.nkpt * self.ndim) if self.use_keypoints else 5), dtype=np.float32)
        if self.use_keypoints:
            keypoints = lb[:, 5:].reshape(-1, self.nkpt, self.ndim)
            if self.ndim == 2:
                kpt_mask = np.where((keypoints[..., 0] < 0) | (keypoints[..., 1] < 0), 0.0, 1.0).astype(np.float32)
                keypoints = np.concatenate([keypoints, kpt_mask[..., None]], axis=-1)  # (nl, nkpt, 3)
        lb = lb[:, :5]
        return lb, segments, keypoints

    def label_format(self, label: tuple) -> dict[str, Any] | None:
        im_file, ir_file, lb, shape, segments, keypoint = label
        if im_file and ir_file:
            return {
                "im_file": im_file,
                "ir_file": ir_file,
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
        self.ir_files = [self.ir_files[i] for i in irect]
        if isinstance(self.labels, list):
            self.labels = [self.labels[i] for i in irect]
        elif isinstance(self.labels, dict):
            self.labels = {lb: [self.labels[lb][i] for i in irect] for lb in self.label_names}
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
                    for modal in self.modal_names:
                        try:
                            b += self.data_files[modal][i].stat().st_size
                        except FileNotFoundError:
                            b += 0.0
                else:
                    if x:
                        for modal in self.modal_names:
                            self.data[modal][i], self.hw0[modal][i], self.hw[modal][i] = x[modal]
                            if self.data[modal][i]:
                                b += self.data[modal][i].nbytes
                pbar.desc = f"Caching {self.mode} data ({b / gb:.5f}GB {storage})"
            pbar.close()

    def load_data(self, i: int, rect_mode: bool = True) -> tuple[np.ndarray | None]:
        """Loads 1 img and label from dataset index 'i', returns (img, label)."""
        im, imf, imfn = self.imgs[i], self.img_files[i], self.img_npy_files[i]
        ir, irf, irfn = self.irs[i], self.ir_files[i], self.ir_npy_files[i]

        # load image and ir
        if im is None and ir is None:  # not cached in RAM
            if imfn.exists():  # cached in Disk
                try:
                    im = np.load(imfn)
                except Exception as e:
                    LOGGER.warning(f"Removing corrupt *.npy image file {imfn} due to: {e}.")
                    Path(imfn).unlink(missing_ok=True)
                    im = self.file_read(imf)
            else:
                im = self.file_read(imf)

            if irfn.exists():  # cached in Disk
                try:
                    ir = np.load(irfn)
                except Exception as e:
                    LOGGER.warning(f"Removing corrupt *.npy image file {irfn} due to: {e}.")
                    Path(irfn).unlink(missing_ok=True)
                    ir = self.file_read(irf)
            else:
                ir = self.file_read(irf)

            if im is not None and ir is not None:
                im_h0, im_w0 = im.shape[:2]  # orig hw
                ir_h0, ir_w0 = ir.shape[:2]
                if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
                    im_r = self.imgsz / max(im_h0, im_w0)  # ratio
                    if im_r != 1:  # if sizes are not equal
                        w, h = (min(math.ceil(im_w0 * im_r), self.imgsz), min(math.ceil(im_h0 * im_r), self.imgsz))
                        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
                    ir_r = self.imgsz / max(ir_h0, ir_w0)
                    if ir_r != 1:  # if sizes are not equal
                        w, h = (min(math.ceil(ir_w0 * ir_r), self.imgsz), min(math.ceil(ir_h0 * ir_r), self.imgsz))
                        ir = cv2.resize(ir, (w, h), interpolation=cv2.INTER_LINEAR)
                else:
                    if not (im_h0 == im_w0 == self.imgsz):  # resize by stretching image to square imgsz
                        im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
                    if not (ir_h0 == ir_w0 == self.imgsz):  # resize by stretching image to square imgsz
                        ir = cv2.resize(ir, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

                # Add to buffer if training with augmentations
                if self.augment:
                    self.imgs[i], self.hw0["im"][i], self.hw["im"][i] = im, (im_h0, im_w0), im.shape[:2]
                    self.irs[i], self.hw0["ir"][i], self.hw["ir"][i] = ir, (ir_h0, ir_w0), ir.shape[:2]
                    self.augment_buffer.append(i)
                    if 1 < len(self.augment_buffer) >= self.max_augment_buffer_length:  # prevent empty buffer
                        j = self.augment_buffer.pop(0)
                        if self.cache != "ram":
                            self.imgs[j], self.hw0["im"][j], self.hw["im"][j] = None, None, None
                            self.irs[j], self.hw0["ir"][j], self.hw["ir"][j] = None, None, None

                return {"img": (im, (im_h0, im_w0), im.shape[:2]), "ir": (ir, (ir_h0, ir_w0), ir.shape[:2])}
            else:
                self.corrupt_idx.add(i)  # data corrupt
                return {"img": (None, None, None), "ir": (None, None, None)}

        return {"img": (im, self.hw0["im"][i], self.hw["im"][i]), "ir": (ir, self.hw0["ir"][i], self.hw["ir"][i])}

    def __getitem__(self, index) -> dict[str, Any]:
        sample = self.get_data_and_label(index)
        if self.transforms:
            sample = self.transforms(sample)  # The transform must take label as input.
        return sample

    def get_data_and_label(self, index: int) -> dict[str, Any]:
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
        return self.update_labels_info(sample)

    def update_labels_info(self, sample: dict[str, Any]) -> dict[str, Any]:
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
            for modal in self.modal_names:
                data.append(self.file_read(self.data_files[modal][i]))
            if all(dt is None for dt in data):
                skips += 1
                continue
            b += np.sum(
                [d.nbytes * (self.imgsz / max(d.shape[0], d.shape[1])) ** 2 if d is not None else 0.0 for d in data]
            )
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

        self.hw0 = {modal: [None] * length for modal in self.modal_names}
        self.register_buffer("hw0", self.hw0)
        self.hw = {modal: [None] * length for modal in self.modal_names}
        self.register_buffer("hw", self.hw)

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
