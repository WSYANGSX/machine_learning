from typing import Any, Optional, Union, Literal

import cv2
import math
import numpy as np
from pathlib import Path

from machine_learning.utils.logger import LOGGER
from machine_learning.types.aliases import FilePath
from machine_learning.dataset.base import DatasetBase


class YoloDataset(DatasetBase):
    """
    YoloDataset class for loading object detection and/or segmentation labels in YOLO format.

    Args:
        imgs (list[str] | np.ndarray): Path list to imgs or imgs itself with np.ndarray format.
        labels (list[str] | np.ndarray): Path list to the labels or labels itself with np.ndarray format.
        imgsz (list[str] | np.ndarray): Image size. Defaults to 640.
        rect (bool, optional): If True, rectangular training is used. Defaults to False.
        stride (int, optional): Stride. Defaults to 32.
        pad (float, optional): Padding. Defaults to 0.0.
        single_cls (bool, optional): If True, single class training is used. Defaults to False.
        classes (list): List of included classes. Default is None.
        cache (bool, optional): Cache data to RAM or disk during training. Defaults to False.
        augment (bool, optional): If True, data augmentation is applied. Defaults to True.
        hyp (dict, optional): Hyperparameters to apply data augmentation. Defaults to None.
        batch_size (int, optional): Size of batches. Defaults to None.
        fraction (float): Fraction of dataset to utilize. Default is 1.0 (use all data).

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
    """

    def __init__(
        self,
        imgs: np.ndarray | list[FilePath],
        labels: np.ndarray | list[FilePath],
        imgsz: int = 640,
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
            mode=mode
        )  # cache imgs and labels

        if self.rect:
            assert self.batch_size is not None
            self.set_rectangle()

    def label_format(self, label):
        return super().label_format(label)

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

    def load(self, i: int) -> tuple[Union[np.ndarray, None], Union[np.ndarray, None]]:
        """Loads 1 img and label from dataset index 'i', returns (img, label)."""
        dt, dtf, dtfn = self.data[i], self.data_files[i], self.data_npy_files[i]
        lb, lbf, lbfn = self.labels[i], self.label_files[i], self.label_npy_files[i]

        # load data
        if dt is not None:  # cached in RAM
            data = dt
        elif dtfn.exists():  # cached in Disk
            try:
                data = np.load(dtfn)
            except Exception as e:
                LOGGER.warning(f"Removing corrupt *.npy image file {dtfn} and {lbfn} due to: {e}")
                Path(dtfn).unlink(missing_ok=True)
                data = self.fread(dtf)
        else:
            data = self.fread(dtf)

        # load label
        if lb is not None:  # cached in RAM
            label = lb
        elif lbfn.exists():  # cached in Disk
            try:
                label = np.load(lbfn)
            except Exception as e:
                LOGGER.warning(f"Removing corrupt *.npy image file {dtfn} and {lbfn} due to: {e}")
                Path(lbfn).unlink(missing_ok=True)
                label = self.lread(lbf)
        else:
            label = self.lread(lbf)

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
            self.data[i] = data
            self.labels[i] = label  # load to buffer for faster augment
            self.buffer.append(i)
            if len(self.buffer) >= self.max_buffer_length:  # prevent empty buffer
                j = self.buffer.pop(0)
                if self.cache != "ram":
                    self.data[j], self.labels[j] = None, None

        return data, label

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

    def check_data_cache_ram(self, safety_margin=0.5):
        return super().check_data_cache_ram(safety_margin)

    def check_data_cache_disk(self, safety_margin=0.5):
        return super().check_data_cache_disk(safety_margin)

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
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,  # only affect training.
            )
        )
        return transforms

    def close_mosaic(self, hyp):
        """Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations."""
        hyp.mosaic = 0.0  # set mosaic ratio=0.0
        hyp.copy_paste = 0.0  # keep the same behavior as previous v8 close-mosaic
        hyp.mixup = 0.0  # keep the same behavior as previous v8 close-mosaic
        self.transforms = self.build_transforms(hyp)


class MultimodalDataset(DatasetBase):
    def __init__(self):
        super().__init__()
        pass
