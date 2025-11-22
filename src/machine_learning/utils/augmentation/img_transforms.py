"""Image transform and augmentation module

Adapted from Ultralytics YOLO base dataset implementation.
Source: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/augment.py.
"""

from __future__ import annotations
from typing import Any, Callable, Literal

import cv2
import math
import torch
import random
import numpy as np
import albumentations as A

from PIL import Image
from copy import deepcopy
from torch.utils.data import Dataset


from ultralytics.utils.metrics import bbox_ioa
from ultralytics.utils.instance import Instances
from ultralytics.utils.ops import segment2box, xyxyxyxy2xywhr
from ultralytics.data.utils import polygons2masks, polygons2masks_overlap
from ultralytics.utils.torch_utils import TORCHVISION_0_10, TORCHVISION_0_11, TORCHVISION_0_13

from .core import TransformInterface, Compose
from .utils import ensure_contiguous_output, masks_to_overlap
from machine_learning.utils.logger import LOGGER


DEFAULT_MEAN = (0.0, 0.0, 0.0)
DEFAULT_STD = (1.0, 1.0, 1.0)
DEFAULT_CROP_FRACTION = 1.0


# Base image augmentations classes--------------------------------------------------------------------------------------
class TransformBase(TransformInterface):
    """
    Base class for image transformations interface.

    This class serves as a foundation for implementing various image processing operations, designed to be
    compatible with classification、detection and segmentation tasks.
    """

    _targets = ("img", "ir", "depth")
    _annotations = ("instances", "mask", "masks")

    def __init__(
        self,
        p: float = 1,
        pre_transform: "TransformBase" | Compose = None,
    ) -> None:
        """
        Initializes the ImgTransformBase object.

        This constructor sets up the base transformation object, which can be extended for specific image
        processing tasks. It is designed to be compatible with both classification and semantic segmentation.

        Args:
            p (float): The probability for the transform to be applied.
        """
        super().__init__(p, pre_transform=pre_transform)

    @property
    def targets(self) -> dict[str, Callable]:
        """Get mapping of target keys to their corresponding processing functions."""
        targets = {}
        # data targets
        for target in self._targets:
            targets[target] = self.apply_to_target
        # annotation targets
        targets.update(
            {
                "instances": self.apply_to_instances,
                "mask": self.apply_to_mask,
                "masks": self.apply_to_masks,
            }
        )
        return targets

    def get_params_on_sample(self, sample: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        return {}

    def apply_to_target(self, target: np.ndarray, category: str, **params: dict[str, Any]) -> np.ndarray:
        """
        Applies transformations to a target in self._targets.

        This method is intended to be overridden by subclasses to implement specific image transformation
        logic. In its base form, it returns the input sample unchanged.

        Args:
            target (np.ndarray):  A target in self._targets with shape (H, W) or (H, W, 3), e.g. image, infrared, depth.
            category (str | None): The name of the target, e.g. "img", "image", "ir", "depth".
            params (dict[str, Any]): Additional parameters for the transformation.
        """
        return target

    def apply_to_instances(self, instances: Instances, **params: dict[str, Any]) -> Instances:
        """
        Applies transformations to object instances.

        This method is responsible for applying various transformations to object instances within the given
        sample. It is designed to be overridden by subclasses to implement specific instance transformation
        logic.

        Args:
            instances (Instances):  Container for bounding boxes, segments, and keypoints of detected objects.
            params (dict[str, Any]): Additional parameters for the transformation.
        """
        return instances

    def apply_to_mask(self, mask: np.ndarray, **params: dict[str, Any]) -> np.ndarray:
        """
        Applies semantic mask transformations.

        All mask inputs are required to be (H, W) for semantic segmentation.

        Args:
            mask (np.ndarray): The mask used for semantic segmentation tasks with shape (H, W).
            params (dict[str, Any]): Additional parameters for the transformation.
        """
        return mask

    def apply_to_masks(self, masks: np.ndarray, **params: dict[str, Any]) -> np.ndarray:
        """
        Applies instances masks transformations.

        All mask inputs are required to be (N, H, W) for instances egmentation.
        Args:
            masks (np.ndarray): The masks used for instances egmentation tasks with shape (N, H, W).
            params (dict[str, Any]): Additional parameters for the transformation.
        """
        if masks.size == 0:
            fake_mask = np.zeros(params["size0"], dtype=masks.dtype)
            fake_mask = self.apply_to_mask(fake_mask, **params)
            return np.zeros((0, *fake_mask.shape[:2]), dtype=masks.dtype)

        return np.stack([self.apply_to_mask(mask, **params) for mask in masks])

    def apply_with_params(self, sample: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        """Apply data augmentation to sample with parameters."""
        res = {}

        for key, val in sample.items():
            if key in self.key2func and val is not None:
                target_function = self.key2func[key]

                if key in self._targets:
                    extend_params = {"category": key, **params}
                    res[key] = ensure_contiguous_output(target_function(ensure_contiguous_output(val), **extend_params))
                else:
                    res[key] = ensure_contiguous_output(target_function(ensure_contiguous_output(val), **params))
            else:
                res[key] = val

        return self.update_sample(res, **params)

    def update_sample(self, sample: dict[str, Any], **params: Any) -> dict[str, Any]:
        """Update the logic for annotations in sample, to be implemented by subclasses."""
        return sample

    def get_target_size(self, sample: dict[str, Any]) -> tuple[int, int]:
        """Get the target size (h, w) of targets in a sample."""
        if "resized_shape" in sample:
            size = sample["resized_shape"][:2]
        else:
            size = next(
                (sample[t].shape[:2] for t in self._targets if t in sample and sample[t] is not None),
                None,
            )
            if size is None:
                raise ValueError("No valid target found in the sample to infer data size.")
        return size


class MixTransformBase(TransformBase):
    """
    Base class for image mix transformations like MixUp, Mosaic and CopyPaste.

    This class provides a foundation for implementing mix transformations on datasets. It handles the
    probability-based application of transforms and manages the mixing of multiple images and labels.
    """

    def __init__(
        self,
        dataset: Dataset,
        p: float = 0.0,
        pre_transform: TransformBase | Compose = None,
    ) -> None:
        """
        Initializes the ImgMixTransformBase object for mix transformations like MixUp and Mosaic.

        This class serves as a base for implementing mix transformations in image processing pipelines.

        Args:
            dataset (Any): The dataset object containing images and annotations for mixing.
            p (float): Probability of applying the mix transformation. Should be in the range [0.0, 1.0].
            pre_transform (Callable | None): Optional transform to apply before mixing.
        """
        super().__init__(p=p, pre_transform=pre_transform)
        self.dataset = dataset

    def get_params(self) -> dict[str, Any]:
        """
        Obtain transformation parameters independent of input to ensure consistency among different fields.
        """
        # Get index of one or three other images
        indexes = self.get_indexes()
        if isinstance(indexes, int):
            indexes = [indexes]

        # Get images information will be used for
        mix_samples = [self.dataset.get_sample(i) for i in indexes]
        if self.pre_transform is not None:
            for i, data in enumerate(mix_samples):
                mix_samples[i] = self.pre_transform(data)
        assert len(mix_samples), "There are no other images for mosaic augment."

        return {"mix_samples": mix_samples}

    def get_indexes(self) -> int:
        """
        Gets index for mix augmentation.
        """
        return random.randint(0, len(self.dataset) - 1)

    def apply_to_target(
        self,
        target: np.ndarray,
        category: str,
        mix_samples: list[dict[str, Any]],
        **params: dict[str, Any],
    ) -> np.ndarray:
        """
        Applies image transformations, such as MixUp or Mosaic.
        """
        raise NotImplementedError

    def apply_to_instances(
        self, instances: Instances, mix_samples: list[dict[str, Any]], **params: dict[str, Any]
    ) -> Instances:
        """
        Applies transformations to instances (e.g., bboxes, keypoints, etc.).
        """
        raise NotImplementedError

    def apply_to_mask(
        self, mask: np.ndarray, mix_samples: list[dict[str, Any]], **params: dict[str, Any]
    ) -> np.ndarray:
        """
        Applies semantic segmentation transformations.
        """
        raise NotImplementedError

    def apply_to_masks(
        self, masks: np.ndarray, mix_samples: list[dict[str, Any]], **params: dict[str, Any]
    ) -> np.ndarray:
        """
        Applies instance segmentation transformations.
        """
        raise NotImplementedError

    def __call__(self, sample):
        """Apply the transform to the input sample."""
        params = self.get_params()
        params_dependent_on_data = self.get_params_on_sample(sample, params)
        params.update(params_dependent_on_data)
        self._params = params

        sample = self._sort_sample_keys(sample)  # for process_params pass correctly
        if random.random() < self.p:
            return self.apply_with_params(sample, params)
        return sample


class Mosaic(MixTransformBase):
    """
    Mosaic augmentation for image datasets.

    This class performs mosaic augmentation by combining multiple (4 or 9) images into a single mosaic image.
    The augmentation is applied to a dataset with a given probability.

    Attributes:
        dataset: The dataset on which the mosaic augmentation is applied.
        imgsz (int): Image size (height and width) after mosaic pipeline of a single image.
        p (float): Probability of applying the mosaic augmentation. Must be in the range 0-1.
        n (int): The grid size, either 4 (for 2x2) or 9 (for 3x3).
        border (Tuple[int, int]): Border size for width and height.
    """

    def __init__(
        self,
        dataset: Dataset,
        imgsz: int = 640,
        p: float = 1.0,
        n: int = 4,
    ):
        """
        Initializes the Mosaic augmentation object.

        This class performs mosaic augmentation by combining multiple (4 or 9) images into a single mosaic image.
        The augmentation is applied to a dataset with a given probability.

        Args:
            dataset (Any): The dataset on which the mosaic augmentation is applied.
            imgsz (int): Image size (height and width) after mosaic pipeline of a single image.
            p (float): Probability of applying the mosaic augmentation. Must be in the range 0-1.
            n (int): The grid size, either 4 (for 2x2) or 9 (for 3x3).
        """
        assert 0 <= p <= 1.0, f"The probability should be in range [0, 1], but got {p}."
        assert n in {4, 9}, "Grid must be equal to 4 or 9."
        super().__init__(dataset=dataset, p=p)
        self.n = n
        self.imgsz = imgsz
        self.border = (-imgsz // 2, -imgsz // 2)  # width, height
        self.buffer_enabled = hasattr(self.dataset, "mosaic_buffer") and len(self.dataset.mosaic_buffer) > 0

    def get_params_on_sample(self, sample: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        return {"size0": self.get_target_size(sample)}

    def get_params(self) -> dict[str, Any]:
        params = super().get_params()
        yc = int(random.uniform(-self.border[0], 2 * self.imgsz + self.border[0]))
        xc = int(random.uniform(-self.border[1], 2 * self.imgsz + self.border[1]))
        params.update({"mosaic_center": (yc, xc)})
        params["areas"] = None  # to record the area of each image in the mosaic
        return params

    def get_indexes(self):
        """
        Returns a list of random indexes from the dataset for mosaic augmentation.

        This method selects random image indexes either from a buffer or from the entire dataset, depending on
        the 'buffer' parameter. It is used to choose images for creating mosaic augmentations.
        """
        if self.buffer_enabled:  # select images from buffer
            return random.choices(list(self.dataset.mosaic_buffer), k=self.n - 1)
        else:  # select any images
            return [random.randint(0, len(self.dataset) - 1) for _ in range(self.n - 1)]

    def apply_with_params(self, sample: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        assert sample.get("rect_shape", None) is None, "rect and mosaic are mutually exclusive."
        return super().apply_with_params(sample, params)

    def apply_to_target(
        self,
        target: np.ndarray,
        category: str,
        mix_samples: list[dict[str, Any]],
        size0: tuple[int, int],
        mosaic_center: tuple[int, int],
        **params: dict[str, Any],
    ) -> np.ndarray:
        # channel
        ch = 3 if category == "img" else 1
        # data type
        pad_val = 0.0 if target.dtype.kind == "f" else 114 if category == "img" else 0

        # mosaic4
        if self.n == 4:
            yc, xc = mosaic_center
            if ch == 1:
                target4 = np.full((self.imgsz * 2, self.imgsz * 2), pad_val, dtype=target.dtype)
            else:
                target4 = np.full((self.imgsz * 2, self.imgsz * 2, ch), pad_val, dtype=target.dtype)

            areas = []
            for i in range(4):
                # Load image
                target = target if i == 0 else mix_samples[i - 1][category]
                h, w = size0 if i == 0 else target.shape[:2]

                # Place img in img4
                if i == 0:  # top left
                    x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                    x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
                elif i == 1:  # top right
                    x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, self.imgsz * 2), yc
                    x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
                elif i == 2:  # bottom left
                    x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(self.imgsz * 2, yc + h)
                    x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
                elif i == 3:  # bottom right
                    x1a, y1a, x2a, y2a = xc, yc, min(xc + w, self.imgsz * 2), min(self.imgsz * 2, yc + h)
                    x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

                if ch == 3 and target.ndim == 2:  # There are grayscale images.
                    target4[y1a:y2a, x1a:x2a] = target[y1b:y2b, x1b:x2b, None]
                else:
                    target4[y1a:y2a, x1a:x2a] = target[y1b:y2b, x1b:x2b]

                areas.append((y2a - y1a) * (x2a - x1a))

            if params["areas"] is None:
                params["areas"] = areas

            return target4

        # mosaic9
        elif self.n == 9:
            hp, wp = -1, -1  # height, width previous
            if ch == 1:
                target9 = np.full((self.imgsz * 3, self.imgsz * 3), pad_val, dtype=target.dtype)
            else:
                target9 = np.full((self.imgsz * 3, self.imgsz * 3, ch), pad_val, dtype=target.dtype)

            areas = []
            crop_x1, crop_y1, crop_x2, crop_y2 = (
                -self.border[1],
                -self.border[0],
                self.imgsz * 3 + self.border[1],
                self.imgsz * 3 + self.border[0],
            )

            for i in range(9):
                target = target if i == 0 else mix_samples[i - 1][category]
                h, w = size0 if i == 0 else target.shape[:2]

                # Place img in img9
                if i == 0:  # center
                    h0, w0 = h, w
                    c = self.imgsz, self.imgsz, self.imgsz + w, self.imgsz + h  # xmin, ymin, xmax, ymax
                elif i == 1:  # top
                    c = self.imgsz, self.imgsz - h, self.imgsz + w, self.imgsz
                elif i == 2:  # top right
                    c = self.imgsz + wp, self.imgsz - h, self.imgsz + wp + w, self.imgsz
                elif i == 3:  # right
                    c = self.imgsz + w0, self.imgsz, self.imgsz + w0 + w, self.imgsz + h
                elif i == 4:  # bottom right
                    c = self.imgsz + w0, self.imgsz + hp, self.imgsz + w0 + w, self.imgsz + hp + h
                elif i == 5:  # bottom
                    c = self.imgsz + w0 - w, self.imgsz + h0, self.imgsz + w0, self.imgsz + h0 + h
                elif i == 6:  # bottom left
                    c = self.imgsz + w0 - wp - w, self.imgsz + h0, self.imgsz + w0 - wp, self.imgsz + h0 + h
                elif i == 7:  # left
                    c = self.imgsz - w, self.imgsz + h0 - h, self.imgsz, self.imgsz + h0
                elif i == 8:  # top left
                    c = self.imgsz - w, self.imgsz + h0 - hp - h, self.imgsz, self.imgsz + h0 - hp

                # c = (x1a, y1a, x2a, y2a)
                x1a, y1a, x2a, y2a = c

                # Intersect with the cropping window
                ix1, iy1, ix2, iy2 = max(x1a, crop_x1), max(y1a, crop_y1), min(x2a, crop_x2), min(y2a, crop_y2)
                iw, ih = max(ix2 - ix1, 0), max(iy2 - iy1, 0)
                areas.append(iw * ih)

                padw, padh = c[:2]
                x1, y1, x2, y2 = max(c[0], 0), max(c[1], 0), min(c[2], self.imgsz * 3), min(c[3], self.imgsz * 3)
                hp, wp = h, w  # Record the current image size for the next use

                if ch == 3 and target.ndim == 2:
                    target9[y1:y2, x1:x2] = target[y1 - padh :, x1 - padw :, None]
                else:
                    target9[y1:y2, x1:x2] = target[y1 - padh :, x1 - padw :]

            if params["areas"] is None:
                params["areas"] = areas

            # Labels assuming imgsz*2 mosaic size
            return target9[-self.border[0] : self.border[0], -self.border[1] : self.border[1]]

    def apply_to_instances(
        self,
        instances: Instances,
        mix_samples: list[dict[str, Any]],
        size0: tuple[int, int],
        mosaic_center: tuple[int, int],
        **params: dict[str, Any],
    ) -> Instances:
        mosaic_instances = []

        # mosaic4
        if self.n == 4:
            yc, xc = mosaic_center  # mosaic center x, y
            for i in range(4):
                if i > 0:
                    instances = mix_samples[i - 1]["instances"]
                h, w = size0 if i == 0 else self.get_target_size(mix_samples[i - 1])

                # Place img in img4
                if i == 0:  # top left
                    x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                    x1b, y1b, _, _ = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
                elif i == 1:  # top right
                    x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, self.imgsz * 2), yc
                    x1b, y1b, _, _ = 0, h - (y2a - y1a), min(w, x2a - x1a), h
                elif i == 2:  # bottom left
                    x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(self.imgsz * 2, yc + h)
                    x1b, y1b, _, _ = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
                elif i == 3:  # bottom right
                    x1a, y1a, x2a, y2a = xc, yc, min(xc + w, self.imgsz * 2), min(self.imgsz * 2, yc + h)
                    x1b, y1b, _, _ = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

                padw, padh = x1a - x1b, y1a - y1b

                instances.convert_bbox(format="xyxy")
                instances.denormalize(w, h)
                instances.add_padding(padw, padh)
                mosaic_instances.append(instances)

        # mosaic9
        elif self.n == 9:
            hp, wp = -1, -1  # height, width previous
            for i in range(9):
                if i > 0:
                    instances = mix_samples[i - 1]["instances"]
                h, w = size0 if i == 0 else self.get_target_size(mix_samples[i - 1])

                # Place img in img9
                if i == 0:  # center
                    h0, w0 = h, w
                    c = self.imgsz, self.imgsz, self.imgsz + w, self.imgsz + h  # xmin, ymin, xmax, ymax
                elif i == 1:  # top
                    c = self.imgsz, self.imgsz - h, self.imgsz + w, self.imgsz
                elif i == 2:  # top right
                    c = self.imgsz + wp, self.imgsz - h, self.imgsz + wp + w, self.imgsz
                elif i == 3:  # right
                    c = self.imgsz + w0, self.imgsz, self.imgsz + w0 + w, self.imgsz + h
                elif i == 4:  # bottom right
                    c = self.imgsz + w0, self.imgsz + hp, self.imgsz + w0 + w, self.imgsz + hp + h
                elif i == 5:  # bottom
                    c = self.imgsz + w0 - w, self.imgsz + h0, self.imgsz + w0, self.imgsz + h0 + h
                elif i == 6:  # bottom left
                    c = self.imgsz + w0 - wp - w, self.imgsz + h0, self.imgsz + w0 - wp, self.imgsz + h0 + h
                elif i == 7:  # left
                    c = self.imgsz - w, self.imgsz + h0 - h, self.imgsz, self.imgsz + h0
                elif i == 8:  # top left
                    c = self.imgsz - w, self.imgsz + h0 - hp - h, self.imgsz, self.imgsz + h0 - hp

                padw, padh = c[:2]
                hp, wp = h, w  # Record the current image size for the next use

                instances.convert_bbox(format="xyxy")
                instances.denormalize(w, h)
                instances.add_padding(padw + self.border[0], padh + self.border[1])
                mosaic_instances.append(instances)

        return Instances.concatenate(mosaic_instances, axis=0)

    def apply_to_mask(
        self,
        mask: np.ndarray,
        mix_samples: list[dict[str, Any]],
        size0: tuple[int, int],
        mosaic_center: tuple[int, int],
        **params: dict[str, Any],
    ) -> np.ndarray:
        """
        Applies semantic segmentation transformations.
        """
        if self.n == 4:
            yc, xc = mosaic_center
            hm = wm = self.imgsz * 2
        else:
            hm = wm = self.imgsz * 3

        mosaic_mask = np.zeros((hm, wm), dtype=mask.dtype)

        if self.n == 4:  # n = 4
            for i in range(4):
                mask = mask if i == 0 else mix_samples[i - 1].get("mask")
                h, w = size0 if i == 0 else mask.shape[:2]

                if i == 0:  # top-left
                    x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                    x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
                elif i == 1:  # top-right
                    x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, wm), yc
                    x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
                elif i == 2:  # bottom-left
                    x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(hm, yc + h)
                    x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
                else:  # bottom-right
                    x1a, y1a, x2a, y2a = xc, yc, min(xc + w, wm), min(hm, yc + h)
                    x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

                mosaic_mask[y1a:y2a, x1a:x2a] = mask[y1b:y2b, x1b:x2b]

            return mosaic_mask

        else:  # n = 9
            hp, wp = -1, -1
            for i in range(9):
                mask = mask if i == 0 else mix_samples[i - 1].get("mask")
                h, w = size0 if i == 0 else mask.shape[:2]

                if i == 0:
                    h0, w0 = h, w
                    c = self.imgsz, self.imgsz, self.imgsz + w, self.imgsz + h
                elif i == 1:
                    c = self.imgsz, self.imgsz - h, self.imgsz + w, self.imgsz
                elif i == 2:
                    c = self.imgsz + wp, self.imgsz - h, self.imgsz + wp + w, self.imgsz
                elif i == 3:
                    c = self.imgsz + w0, self.imgsz, self.imgsz + w0 + w, self.imgsz + h
                elif i == 4:
                    c = self.imgsz + w0, self.imgsz + hp, self.imgsz + w0 + w, self.imgsz + hp + h
                elif i == 5:
                    c = self.imgsz + w0 - w, self.imgsz + h0, self.imgsz + w0, self.imgsz + h0 + h
                elif i == 6:
                    c = self.imgsz + w0 - wp - w, self.imgsz + h0, self.imgsz + w0 - wp, self.imgsz + h0 + h
                elif i == 7:
                    c = self.imgsz - w, self.imgsz + h0 - h, self.imgsz, self.imgsz + h0
                else:
                    c = self.imgsz - w, self.imgsz + h0 - hp - h, self.imgsz, self.imgsz + h0 - hp

                x1a, y1a, x2a, y2a = (max(x, 0) for x in c)
                padw, padh = c[:2]
                hp, wp = h, w

                mosaic_mask[y1a:y2a, x1a:x2a] = mask[y1a - padh : y2a - padh, x1a - padw : x2a - padw]

            return mosaic_mask[-self.border[0] : self.border[0], -self.border[1] : self.border[1]]

    def apply_to_masks(
        self,
        masks: np.ndarray,
        mix_samples: list[dict[str, Any]],
        size0: tuple[int, int],
        mosaic_center: tuple[int, int],
        **params,
    ) -> np.ndarray:
        mosaic_masks = []

        if self.n == 4:
            yc, xc = mosaic_center
            hm = wm = self.imgsz * 2

            for i in range(4):
                curr_masks = masks if i == 0 else mix_samples[i - 1].get("masks")
                if curr_masks.size == 0:  # empty
                    continue

                h, w = size0 if i == 0 else curr_masks.shape[1:3]
                if i == 0:  # top left
                    x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                    x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
                elif i == 1:  # top right
                    x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, wm), yc
                    x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
                elif i == 2:  # bottom left
                    x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(hm, yc + h)
                    x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
                else:  # i == 3, bottom right
                    x1a, y1a, x2a, y2a = xc, yc, min(xc + w, wm), min(hm, yc + h)
                    x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

                for k in range(curr_masks.shape[0]):
                    big_mask = np.zeros((hm, wm), dtype=curr_masks.dtype)
                    big_mask[y1a:y2a, x1a:x2a] = curr_masks[k, y1b:y2b, x1b:x2b]
                    mosaic_masks.append(big_mask)

            if len(mosaic_masks) == 0:
                return np.zeros((0, self.imgsz * 2, self.imgsz * 2), dtype=np.uint8)

            return np.stack(mosaic_masks, axis=0)

        elif self.n == 9:
            hm = wm = self.imgsz * 3
            hp, wp = -1, -1  # height, width previous

            for i in range(9):
                curr_masks = masks if i == 0 else mix_samples[i - 1].get("masks")
                if curr_masks.size == 0:
                    continue

                h, w = size0 if i == 0 else curr_masks.shape[1:3]

                if i == 0:  # center
                    h0, w0 = h, w
                    c = self.imgsz, self.imgsz, self.imgsz + w, self.imgsz + h
                elif i == 1:  # top
                    c = self.imgsz, self.imgsz - h, self.imgsz + w, self.imgsz
                elif i == 2:  # top right
                    c = self.imgsz + wp, self.imgsz - h, self.imgsz + wp + w, self.imgsz
                elif i == 3:  # right
                    c = self.imgsz + w0, self.imgsz, self.imgsz + w0 + w, self.imgsz + h
                elif i == 4:  # bottom right
                    c = self.imgsz + w0, self.imgsz + hp, self.imgsz + w0 + w, self.imgsz + hp + h
                elif i == 5:  # bottom
                    c = self.imgsz + w0 - w, self.imgsz + h0, self.imgsz + w0, self.imgsz + h0 + h
                elif i == 6:  # bottom left
                    c = self.imgsz + w0 - wp - w, self.imgsz + h0, self.imgsz + w0 - wp, self.imgsz + h0 + h
                elif i == 7:  # left
                    c = self.imgsz - w, self.imgsz + h0 - h, self.imgsz, self.imgsz + h0
                else:  # i == 8, top left
                    c = self.imgsz - w, self.imgsz + h0 - hp - h, self.imgsz, self.imgsz + h0 - hp

                x1a, y1a, x2a, y2a = (max(x, 0) for x in c)
                padw, padh = c[:2]
                hp, wp = h, w  # record

                for k in range(curr_masks.shape[0]):
                    big_mask = np.zeros((hm, wm), dtype=curr_masks.dtype)
                    big_mask[y1a:y2a, x1a:x2a] = curr_masks[k, y1a - padh : y2a - padh, x1a - padw : x2a - padw]
                    mosaic_masks.append(big_mask)

            if len(mosaic_masks) == 0:
                return np.zeros((0, self.imgsz * 2, self.imgsz * 2), dtype=np.uint8)

            return np.stack(mosaic_masks, axis=0)[:, -self.border[0] : self.border[0], -self.border[1] : self.border[1]]

    def update_sample(self, sample: dict[str, Any], mix_samples: list[dict[str, Any]], **params) -> dict[str, Any]:
        """
        Concatenates and processes annotations in mixed samples for mosaic augmentation.
        """
        imgsz = self.imgsz * 2  # mosaic imgsz
        sample["resized_shape"] = (imgsz, imgsz)
        sample["mosaic_border"] = self.border

        # clip and clean
        if "instances" in sample and "cls" in sample:  # yolo task
            cls = [sample["cls"]]
            for ms in mix_samples:
                cls.append(ms["cls"])
            # update sample
            sample["cls"] = np.concatenate(cls, 0)
            sample["instances"].clip(imgsz, imgsz)
            good = sample["instances"].remove_zero_area_boxes()
            sample["cls"] = sample["cls"][good]

            if "masks" in sample:
                sample["masks"] = sample["masks"][good]

        if "instances" not in sample and "cls" in sample:  # instance segment task
            if "masks" in sample:
                # update cls
                cls = [sample["cls"]]
                for ms in mix_samples:
                    cls.append(ms["cls"])
                sample["cls"] = np.concatenate(cls, 0)
                # update masks
                masks: np.ndarray = sample["masks"]
                if masks.size > 0:
                    indices = masks.sum(axis=(1, 2)) > 0
                    sample["masks"] = masks[indices]
                    sample["cls"] = sample["cls"][indices]

            else:  # classify task
                # classification task with soft labels (Mixup/CutMix-style)
                # cls is expected to be one-hot or probability vector
                if params.get("areas", None) is None:
                    # Revert to simple average weights
                    num = 1 + len(mix_samples)
                    weights = np.full(num, 1.0 / num, dtype=np.float32)
                else:
                    areas = np.array(params["areas"], dtype=np.float32)
                    weights = areas / areas.sum()
                cls = sample["cls"] * weights[0]
                for i, ms in enumerate(mix_samples):
                    cls += ms["cls"] * weights[i + 1]
                sample["cls"] = cls

        return sample


# TODO
class MixUp(MixTransformBase):
    """
    Applies MixUp augmentation to image datasets.

    This class implements the MixUp augmentation technique as described in the paper "mixup: Beyond Empirical Risk
    Minimization" (https://arxiv.org/abs/1710.09412). MixUp combines two images and their annotations using a random
    weight.

    Attributes:
        dataset (Any): The dataset to which MixUp augmentation will be applied.
        p (float): Probability of applying MixUp augmentation. Must be in the range [0, 1].
        alpha (float): Control parameters of the beta distribution.
        pre_transform (Callable | None): Optional transform to apply before MixUp.
    """

    def __init__(
        self,
        dataset: Dataset,
        p: float = 0.0,
        alpha: float = 32.0,
        pre_transform: TransformBase | Compose = None,
    ) -> None:
        """
        Initializes the MixUp augmentation object.

        MixUp is an image augmentation technique that combines two images by taking a weighted sum of their pixel
        values and annotations.

        Args:
            dataset (Any): The dataset to which MixUp augmentation will be applied.
            p (float): Probability of applying MixUp augmentation. Must be in the range [0, 1].
            alpha (float): Control parameters of the beta distribution.
            pre_transform (Callable | None): Optional transform to apply to images before MixUp.
        """
        super().__init__(dataset=dataset, p=p, pre_transform=pre_transform)
        self.alpha = alpha

    def get_params_on_sample(self, sample: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        return {"size0": self.get_target_size(sample)}

    def get_params(self):
        params = super().get_params()
        if self.alpha <= 0:
            ratio = 1.0  # no mixup
        else:
            ratio = np.random.beta(self.alpha, self.alpha)  # mixup ratio, alpha=beta=32.0
        params["ratio"] = ratio
        return params

    def apply_to_target(
        self, target: np.ndarray, category: str, mix_samples: list[dict[str, Any]], **params: dict[str, Any]
    ) -> np.ndarray:
        ratio = params["ratio"]

        if category == "img":
            target2 = mix_samples[0][category]
            if target.ndim == 2:  # gray
                target = np.repeat(target[..., None], 3, axis=2)
            if target2.ndim == 2:  # gray
                target2 = np.repeat(target2[..., None], 3, axis=2)
            return (target * ratio + target2 * (1 - ratio)).astype(np.uint8)

        else:
            return target

    def apply_to_instances(
        self, instances: Instances, mix_samples: list[dict[str, Any]], **params: dict[str, Any]
    ) -> Instances:
        return Instances.concatenate([instances, mix_samples[0]["instances"]], axis=0)

    def update_sample(
        self, sample: dict[str, Any], mix_samples: list[dict[str, Any]], **params: dict[str, Any]
    ) -> dict[str, Any]:
        sample["cls"] = np.concatenate([sample["cls"], mix_samples[0]["cls"]], 0)
        return sample

    def apply_with_params(self, sample: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        sample2 = params["mix_samples"][0]
        if params["size0"] != self.get_target_size(sample2):
            return sample  # skip mixup if sizes are different
        else:
            return super().apply_with_params(sample, params)


class CutMix(MixTransformBase):
    """Apply CutMix augmentation to image datasets as described in the paper https://arxiv.org/abs/1905.04899.

    CutMix combines two images by replacing a random rectangular region of one image with the corresponding region from
    another image, and adjusts the labels proportionally to the area of the mixed region.

    Attributes:
        dataset (Any): The dataset to which CutMix augmentation will be applied.
        pre_transform (Callable | None): Optional transform to apply before CutMix.
        p (float): Probability of applying CutMix augmentation.
        beta (float): Beta distribution parameter for sampling the mixing ratio.
        num_areas (int): Number of areas to try to cut and mix.
    """

    def __init__(
        self,
        dataset: Dataset,
        p: float = 0.0,
        beta: float = 1.0,
        num_areas: int = 3,
        pre_transform: TransformBase | Compose = None,
    ) -> None:
        """Initialize the CutMix augmentation object.

        Args:
            dataset (Any): The dataset to which CutMix augmentation will be applied.
            pre_transform (Callable | None): Optional transform to apply before CutMix.
            p (float): Probability of applying CutMix augmentation.
            beta (float): Beta distribution parameter for sampling the mixing ratio.
            num_areas (int): Number of areas to try to cut and mix.
        """
        super().__init__(dataset=dataset, p=p, pre_transform=pre_transform)
        self.beta = beta
        self.num_areas = num_areas

    def get_params(self):
        params = super().get_params()
        params["process_params"] = {}
        return params

    def get_params_on_sample(self, sample: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        """
        Generates target size0 and random bounding box coordinates for CutMix augmentation.
        """

        def _rand_bbox(width: int, height: int) -> tuple[int, int, int, int]:
            """Generate random bounding box coordinates for the cut region.

            Args:
                width (int): Width of the image.
                height (int): Height of the image.
            """
            # Sample mixing ratio from Beta distribution
            lam = np.random.beta(self.beta, self.beta)

            cut_ratio = np.sqrt(1.0 - lam)
            cut_w = int(width * cut_ratio)
            cut_h = int(height * cut_ratio)

            # Random center
            cx = np.random.randint(width)
            cy = np.random.randint(height)

            # Bounding box coordinates
            x1 = np.clip(cx - cut_w // 2, 0, width)
            y1 = np.clip(cy - cut_h // 2, 0, height)
            x2 = np.clip(cx + cut_w // 2, 0, width)
            y2 = np.clip(cy + cut_h // 2, 0, height)

            return x1, y1, x2, y2

        h, w = size0 = self.get_target_size(sample)
        length = len(sample["instances"].segments) if "instances" in sample else None
        cut_areas = np.asarray([_rand_bbox(w, h) for _ in range(self.num_areas)], dtype=np.float32)

        valid_idxs = np.arange(self.num_areas)
        if "instances" in sample:
            # Ensure at least one area has no significant overlap with existing boxes
            ioa = bbox_ioa(cut_areas, sample["instances"].bboxes)  # (self.num_areas, num_boxes)
            valid_idxs = np.nonzero(ioa.sum(axis=1) <= 0)[0]

        return {"size0": size0, "cut_areas": cut_areas, "length": length, "valid_idxs": valid_idxs}

    def apply_to_target(
        self, target: np.ndarray, category: str, mix_samples: list[dict[str, Any]], **params: dict[str, Any]
    ) -> np.ndarray:
        if len(params["valid_idxs"]) == 0:
            return target

        area = params["cut_areas"][np.random.choice(params["valid_idxs"])]
        params["process_params"]["area"] = area
        sample2 = mix_samples[0]
        target2: np.ndarray = sample2[category]
        x1, y1, x2, y2 = area.astype(np.int32)

        # img gray to 3 channels
        if category == "img":
            if target.ndim == 2:
                target = np.repeat(target[..., None], 3, axis=2)
            if target2.ndim == 2:
                target2 = np.repeat(target2[..., None], 3, axis=2)

        if "instances" not in sample2:  # no bboxes limit
            # classification task、 mask/masks segment task
            target[y1:y2, x1:x2] = target2[y1:y2, x1:x2]
            return target

        else:  # bboxes limit
            # yolo task
            ioa2 = bbox_ioa(np.array(area)[None], sample2["instances"].bboxes).squeeze(0)
            indexes2 = np.nonzero(ioa2 >= (0.01 if params["length"] else 0.1))[0]
            params["process_params"]["indexes2"] = indexes2
            if len(indexes2) == 0:
                return target
            target[y1:y2, x1:x2] = target2[y1:y2, x1:x2]
            return target

    def apply_to_instances(
        self, instances: Instances, mix_samples: list[dict[str, Any]], **params: dict[str, Any]
    ) -> Instances:
        if len(params["valid_idxs"]) == 0:
            return instances

        sample2 = mix_samples[0]
        area = params["process_params"]["area"]

        indexes2 = params["process_params"]["indexes2"]
        if len(indexes2) == 0:
            return instances

        instances2 = sample2["instances"][indexes2]
        instances2.convert_bbox("xyxy")
        h, w = params["size0"]
        instances2.denormalize(w, h)

        # Restrain instances2 to the random bounding border
        x1, y1, x2, y2 = area.astype(np.int32)
        instances2.add_padding(-x1, -y1)
        instances2.clip(x2 - x1, y2 - y1)
        instances2.add_padding(x1, y1)

        return Instances.concatenate([instances, instances2], axis=0)

    def apply_to_mask(
        self, mask: np.ndarray, mix_samples: list[dict[str, Any]], **params: dict[str, Any]
    ) -> np.ndarray:
        """
        Applies CutMix augmentation to semantic segmentation masks.
        """
        if len(params["valid_idxs"]) == 0:
            return mask

        area = params["process_params"]["area"]
        sample2 = mix_samples[0]
        mask2 = sample2["mask"]
        x1, y1, x2, y2 = area.astype(np.int32)

        if "instances" not in sample2:  # no bboxes limit
            # classification task、 mask/masks segment task
            patch = mask2[y1:y2, x1:x2]
            mask[y1:y2, x1:x2] = patch
            return mask

        else:  # bboxes limit
            # yolo task
            indexes2 = params["process_params"]["indexes2"]
            if len(indexes2) == 0:
                return mask
            mask[y1:y2, x1:x2] = mask2[y1:y2, x1:x2]
            return mask

    def apply_to_masks(
        self, masks: np.ndarray, mix_samples: list[dict[str, Any]], **params: dict[str, Any]
    ) -> np.ndarray:
        """
        Applies CutMix augmentation to instance masks, masks shape: (N, H, W)
        """
        if len(params["valid_idxs"]) == 0:
            return masks

        sample2 = mix_samples[0]
        masks2 = sample2["masks"]
        area = params["process_params"]["area"]
        x1, y1, x2, y2 = area.astype(np.int32)

        if masks2.size == 0:  # no instances in masks2
            return masks

        if "instances" not in sample2:  # no bboxes limit
            # classification task、 mask/masks segment task
            inside2 = masks2[:, y1:y2, x1:x2].sum(axis=(1, 2)) > 0
            sel2 = masks2[inside2]
            sel2[:, :y1, :x1], sel2[:, y2:, x2:] = 0, 0  # clear outside the patch
            masks[:, y1:y2, x1:x2] = 0  # clear the patch area
            inside1 = masks.sum(axis=(1, 2)) > 0
            sel1 = masks[inside1]

            params["process_params"]["inside1"] = inside1
            params["process_params"]["inside2"] = inside2
            combined = np.concatenate([sel1, sel2], axis=0)

        else:
            # yolo task
            indexes2 = params["process_params"]["indexes2"]
            if len(indexes2) == 0:
                return masks

            # Which masks have pixels within the patch
            masks2 = masks2[indexes2]
            inside2 = masks2[:, y1:y2, x1:x2].sum(axis=(1, 2)) > 0
            sel2 = masks2[inside2]
            sel2[:, :y1, :x1], sel2[:, y2:, x2:] = 0, 0  # clear outside the patch
            masks[:, y1:y2, x1:x2] = 0  # clear the patch area
            inside1 = masks.sum(axis=(1, 2)) > 0
            sel1 = masks[inside1]

            params["process_params"]["inside1"] = inside1
            params["process_params"]["inside2"] = inside2
            combined = np.concatenate([sel1, sel2], axis=0)

        return combined

    def apply_with_params(self, sample: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        mix_sample = params["mix_samples"][0]
        if params["size0"] != self.get_target_size(mix_sample):
            return sample  # skip mixup if sizes are different
        else:
            return super().apply_with_params(sample, params)

    def update_sample(self, sample: dict[str, Any], mix_samples: list[dict[str, Any]], **params) -> dict[str, Any]:
        """
        Adjusts the annotations based on the area of the CutMix patch.
        """
        h, w = params["size0"]
        area = params["process_params"]["area"]

        # yolo task
        if "instances" in sample and "cls" in sample:
            indexes2 = params["process_params"]["indexes2"]

            if "masks" in sample:
                inside1 = params["process_params"].get("inside1", None)
                inside2 = params["process_params"].get("inside2", None)

                cls1_new = sample["cls"][inside1] if inside1 is not None else sample["cls"]
                cls2_new = mix_samples[0]["cls"][indexes2][inside2] if inside2 is not None else None
                sample["cls"] = np.concatenate([cls1_new, cls2_new], axis=0) if cls2_new is not None else cls1_new

            else:
                sample["cls"] = np.concatenate([sample["cls"], mix_samples[0]["cls"][indexes2]], axis=0)

        if "instances" not in sample and "cls" in sample:
            if "masks" in sample:
                inside1 = params["process_params"].get("inside1", None)
                inside2 = params["process_params"].get("inside2", None)

                cls1_new = sample["cls"][inside1] if inside1 is not None else sample["cls"]
                cls2_new = mix_samples[0]["cls"][inside2] if inside2 is not None else None
                sample["cls"] = np.concatenate([cls1_new, cls2_new], axis=0) if cls2_new is not None else cls1_new

            else:  # classification
                patch_area = float((area[2] - area[0]) * (area[3] - area[1]))
                alpha = patch_area / float(w * h)
                # Expect cls to be one-hot or probability vectors
                sample["cls"] = sample["cls"] * (1.0 - alpha) + mix_samples[0]["cls"] * alpha
                return sample

        return sample


# TODO
class CopyPaste(MixTransformBase):
    """CopyPaste class for applying Copy-Paste augmentation to image datasets.

    This class implements the Copy-Paste augmentation technique as described in the paper "Simple Copy-Paste is a Strong
    Data Augmentation Method for Instance Segmentation" (https://arxiv.org/abs/2012.07177). It combines objects from
    different images to create new training samples.

    Attributes:
        dataset (Any): The dataset to which Copy-Paste augmentation will be applied.
        pre_transform (Callable | None): Optional transform to apply before Copy-Paste.
        p (float): Probability of applying Copy-Paste augmentation.
    """

    def __init__(
        self,
        dataset: Dataset = None,
        p: float = 0.5,
        mode: Literal["mixup", "flip"] = "flip",
        pre_transform: TransformBase | Compose = None,
    ) -> None:
        """Initialize CopyPaste object with dataset, pre_transform, and probability of applying MixUp."""
        super().__init__(dataset=dataset, p=p, pre_transform=pre_transform)
        assert mode in {"flip", "mixup"}, f"Expected `mode` to be `flip` or `mixup`, but got {mode}."
        self.mode = mode

    def get_params(self):
        """
        Obtain transformation parameters independent of input to ensure consistency among different fields.
        """
        if self.mode == "flip":
            return {}

        else:
            # Get index of another image
            indexes = self.get_indexes()
            if isinstance(indexes, int):
                indexes = [indexes]

            # Get images information will be used for
            mix_samples = [self.dataset.get_sample(i) for i in indexes]
            if self.pre_transform is not None:
                for i, data in enumerate(mix_samples):
                    mix_samples[i] = self.pre_transform(data)
            assert len(mix_samples), "There are no other images for mosaic augment."

            return {"mix_samples": mix_samples}

    def _mix_transform(self, sample: dict[str, Any], mix_samples: list[dict[str, Any]]) -> dict[str, Any]:
        """Apply Copy-Paste augmentation to combine objects from another image into the current image."""
        sample2 = mix_samples[0]
        return self._transform(sample, sample2)

    def apply_with_params(self, sample: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        """Apply Copy-Paste augmentation to an image and its labels."""
        if len(sample["instances"].segments) == 0:
            return sample
        if self.mode == "flip":
            return self._transform(sample)
        sample = self._mix_transform(sample, params["mix_samples"])

        return sample

    def _transform(self, sample1: dict[str, Any], sample2: dict[str, Any] = {}) -> dict[str, Any]:
        """Apply Copy-Paste augmentation to combine objects from another image into the current image."""
        im = sample1["img"]
        if "mosaic_border" not in sample1:
            im = im.copy()  # avoid modifying original non-mosaic image
        cls = sample1["cls"]
        h, w = im.shape[:2]
        instances = sample1.pop("instances")
        instances.convert_bbox(format="xyxy")
        instances.denormalize(w, h)

        im_new = np.zeros(im.shape, np.uint8)
        instances2 = sample2.pop("instances", None)
        if instances2 is None:
            instances2 = deepcopy(instances)
            instances2.fliplr(w)
        ioa = bbox_ioa(instances2.bboxes, instances.bboxes)  # intersection over area, (N, M)
        indexes = np.nonzero((ioa < 0.30).all(1))[0]  # (N, )
        n = len(indexes)
        sorted_idx = np.argsort(ioa.max(1)[indexes])
        indexes = indexes[sorted_idx]
        for j in indexes[: round(self.p * n)]:
            cls = np.concatenate((cls, sample2.get("cls", cls)[[j]]), axis=0)
            instances = Instances.concatenate((instances, instances2[[j]]), axis=0)
            cv2.drawContours(im_new, instances2.segments[[j]].astype(np.int32), -1, (1, 1, 1), cv2.FILLED)

        result = sample2.get("img", cv2.flip(im, 1))  # augment segments
        if result.ndim == 2:  # cv2.flip would eliminate the last dimension for grayscale images
            result = result[..., None]
        i = im_new.astype(bool)
        im[i] = result[i]

        sample1["img"] = im
        sample1["cls"] = cls
        sample1["instances"] = instances
        return sample1


class RandomPerspective(TransformBase):
    """
    Implements random perspective and affine transformations on images and corresponding annotations.

    This class applies random rotations, translations, scaling, shearing, and perspective transformations
    to images and their associated bounding boxes, segments, and keypoints. It can be used as part of an
    augmentation pipeline for object detection and instance segmentation tasks.

    Attributes:
        degrees (float): Maximum absolute degree range for random rotations.
        translate (float): Maximum translation as a fraction of the image size.
        scale (float): Scaling factor range, e.g., scale=0.1 means 0.9-1.1.
        shear (float): Maximum shear angle in degrees.
        perspective (float): Perspective distortion factor.
        border (Tuple[int, int]): Mosaic border size as (x, y).
        pre_transform (Callable | None): Optional transform to apply before the random perspective.
    """

    def __init__(
        self,
        p: float = 1,
        degrees: float = 0.0,
        translate: float = 0.1,
        scale: float = 0.5,
        shear: float = 0.0,
        perspective: float = 0.0,
        border: tuple[int, int] = (0, 0),
        pre_transform: TransformBase | Compose = None,
    ):
        """
        Initializes RandomPerspective object with transformation parameters.

        This class implements random perspective and affine transformations on images and corresponding bounding boxes,
        segments, and keypoints. Transformations include rotation, translation, scaling, and shearing.

        Args:
            degrees (float): Degree range for random rotations.
            translate (float): Fraction of total width and height for random translation.
            scale (float): Scaling factor interval, e.g., a scale factor of 0.5 allows a resize between 50%-150%.
            shear (float): Shear intensity (angle in degrees).
            perspective (float): Perspective distortion factor.
            border (Tuple[int, int]): Tuple specifying mosaic border (top/bottom, left/right).
            pre_transform (Callable | None): Function/transform to apply to the image before starting the random
                transformation.
        """
        super().__init__(p=p, pre_transform=pre_transform)
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.border = border  # mosaic border

    def get_params_on_sample(self, sample: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        size0 = self.get_target_size(sample=sample)

        # border
        border = sample.pop("mosaic_border", self.border)
        dsize = (size0[1] + border[1] * 2, size0[0] + border[0] * 2)  # w, h, desire img size

        # Center
        C = np.eye(3, dtype=np.float32)
        C[0, 2] = -size0[1] / 2  # x translation (pixels)
        C[1, 2] = -size0[0] / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3, dtype=np.float32)
        P[2, 0] = random.uniform(-self.perspective, self.perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-self.perspective, self.perspective)  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3, dtype=np.float32)
        a = random.uniform(-self.degrees, self.degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(1 - self.scale, 1 + self.scale)
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3, dtype=np.float32)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)

        # Translation
        T = np.eye(3, dtype=np.float32)
        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * dsize[0]  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * dsize[1]  # y translation (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT

        return {
            "size0": size0,
            "transform_matrix": M,
            "scale": s,
            "border": border,
            "dsize": dsize,
            "origin_sample": deepcopy(sample),
        }

    def __call__(self, sample):
        """Apply the transform to the input sample."""
        if self.pre_transform is not None and "mosaic_border" not in sample:
            sample = self.pre_transform(sample)
        sample.pop("ratio_pad", None)  # do not need ratio pad

        params = self.get_params()
        params_dependent_on_data = self.get_params_on_sample(sample, params)
        params.update(params_dependent_on_data)
        self._params = params

        sample = self._sort_sample_keys(sample)  # for process_params pass correctly
        if random.random() < self.p:
            return self.apply_with_params(sample, params)
        return sample

    def apply_to_target(
        self,
        target: np.ndarray,
        category: str,
        transform_matrix: np.ndarray,
        border: tuple[int, int],
        dsize: tuple[int, int],
        **params,
    ) -> np.ndarray:
        """
        Applies a sequence of affine transformations centered around the target center.
        """
        # data type
        if np.issubdtype(target.dtype, np.floating):
            border_val = 0.0
        else:
            if category.lower() == "img":
                border_val = (114,) * target.shape[2] if target.ndim == 3 else 114
            else:
                border_val = 0

        if (border[0] != 0) or (border[1] != 0) or (transform_matrix != np.eye(3)).any():  # target changed
            if self.perspective:
                target = cv2.warpPerspective(target, transform_matrix, dsize=dsize, borderValue=border_val)
            else:  # affine
                target = cv2.warpAffine(target, transform_matrix[:2], dsize=dsize, borderValue=border_val)

        return target

    def apply_to_instances(
        self,
        instances: Instances,
        transform_matrix: np.ndarray,
        size0: tuple[int, int],
        dsize: tuple[int, int],
        **params,
    ) -> Instances:
        h0, w0 = size0  # origin img (h,w)

        # Make sure the coord formats are right
        instances.convert_bbox(format="xyxy")
        instances.denormalize(w0, h0)

        # bboxes
        bboxes = self._apply_bboxes(instances.bboxes, transform_matrix)

        # segments
        segments = instances.segments
        if len(segments):
            bboxes, segments = self._apply_segments(segments, transform_matrix, dsize)

        # keypoints
        keypoints = instances.keypoints
        if keypoints is not None:
            keypoints = self._apply_keypoints(keypoints, transform_matrix, dsize)

        new_instances = Instances(bboxes, segments, keypoints, bbox_format="xyxy", normalized=False)
        new_instances.clip(*dsize)  # clip

        return new_instances

    def apply_to_mask(
        self,
        mask: np.ndarray,
        transform_matrix: np.ndarray,
        border: tuple[int, int],
        dsize: tuple[int, int],
        **params: dict[str, Any],
    ) -> np.ndarray:
        """
        Apply perspective/affine transform to semantic masks.
        """
        changed = (border[0] != 0) or (border[1] != 0) or (transform_matrix != np.eye(3)).any()
        if not changed or mask.size == 0:
            return mask

        if self.perspective:
            mask = cv2.warpPerspective(
                mask,
                transform_matrix,
                dsize=dsize,
                borderValue=0,
                flags=cv2.INTER_NEAREST,
            )
        else:  # affine
            mask = cv2.warpAffine(
                mask,
                transform_matrix[:2],
                dsize=dsize,
                borderValue=0,
                flags=cv2.INTER_NEAREST,
            )

        return mask

    def _apply_bboxes(self, bboxes: np.ndarray, M: np.ndarray) -> np.ndarray:
        """
        Apply affine transformation to bounding boxes.
        """
        n = len(bboxes)
        if n == 0:
            return bboxes

        xy = np.ones((n * 4, 3), dtype=bboxes.dtype)
        xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3] if self.perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

        # Create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        return np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1)), dtype=bboxes.dtype).reshape(4, n).T

    def _apply_segments(self, segments: np.ndarray, M: np.ndarray, dsize: tuple[int, int]) -> np.ndarray:
        """
        Apply affine transformations to segments and generate new bounding boxes.
        """
        n, num = segments.shape[:2]
        if n == 0:
            return [], segments

        xy = np.ones((n * num, 3), dtype=segments.dtype)
        segments = segments.reshape(-1, 2)
        xy[:, :2] = segments
        xy = xy @ M.T  # transform
        xy = xy[:, :2] / xy[:, 2:3]
        segments = xy.reshape(n, -1, 2)
        bboxes = np.stack([segment2box(xy, dsize[0], dsize[1]) for xy in segments], 0)
        segments[..., 0] = segments[..., 0].clip(bboxes[:, 0:1], bboxes[:, 2:3])
        segments[..., 1] = segments[..., 1].clip(bboxes[:, 1:2], bboxes[:, 3:4])
        return bboxes, segments

    def _apply_keypoints(self, keypoints: np.ndarray, M: np.ndarray, dsize: tuple[int, int]) -> np.ndarray:
        """
        Applies affine transformation to keypoints.
        """
        n, nkpt = keypoints.shape[:2]
        if n == 0:
            return keypoints
        xy = np.ones((n * nkpt, 3), dtype=keypoints.dtype)
        visible = keypoints[..., 2].reshape(n * nkpt, 1)
        xy[:, :2] = keypoints[..., :2].reshape(n * nkpt, 2)
        xy = xy @ M.T  # transform
        xy = xy[:, :2] / xy[:, 2:3]  # perspective rescale or affine
        out_mask = (xy[:, 0] < 0) | (xy[:, 1] < 0) | (xy[:, 0] > dsize[0]) | (xy[:, 1] > dsize[1])
        visible[out_mask] = 0
        return np.concatenate([xy, visible], axis=-1).reshape(n, nkpt, 3)

    def update_sample(self, sample: dict[str, Any], dsize: tuple[int, int], **params) -> dict[str, Any]:
        """
        Applies random perspective and affine transformations to an image and its associated annotations.
        """
        sample["resized_shape"] = (dsize[1], dsize[0])

        if "instances" in sample and "cls" in sample:
            origin_sample = params["origin_sample"]
            origin_instances = origin_sample["instances"]
            origin_instances.convert_bbox(format="xyxy")
            origin_instances.denormalize(*params["size0"][::-1])
            new_instances = sample["instances"]
            scale = params["scale"]

            # Filter instances
            origin_instances.scale(scale_w=scale, scale_h=scale, bbox_only=True)
            # Make the bboxes have the same scale with new_bboxes
            keep = self.box_candidates(
                box1=origin_instances.bboxes.T,
                box2=new_instances.bboxes.T,
                area_thr=0.01 if len(new_instances.segments) else 0.10,
            )
            sample["instances"] = new_instances[keep]
            sample["cls"] = sample["cls"][keep]

            if "masks" in sample:
                sample["masks"] = sample["masks"][keep]

        return sample

    @staticmethod
    def box_candidates(
        box1: np.ndarray,
        box2: np.ndarray,
        wh_thr: float = 2,
        ar_thr: float = 100,
        area_thr: float = 0.1,
        eps: float = 1e-16,
    ):
        """
        Compute candidate boxes for further processing based on size and aspect ratio criteria.

        This method compares boxes before and after augmentation to determine if they meet specified
        thresholds for width, height, aspect ratio, and area. It's used to filter out boxes that have
        been overly distorted or reduced by the augmentation process.
        """
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
        return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates


class RandomHSV(TransformBase):
    """
    Randomly adjusts the Hue, Saturation, and Value (HSV) channels of an image.

    This class applies random HSV augmentation to images within predefined limits set by hgain, sgain, and vgain.

    Attributes:
        hgain (float): Maximum variation for hue. Range is typically [0, 1].
        sgain (float): Maximum variation for saturation. Range is typically [0, 1].
        vgain (float): Maximum variation for value. Range is typically [0, 1].
    """

    def __init__(self, hgain: float = 0.5, sgain: float = 0.5, vgain: float = 0.5) -> None:
        """
        Initializes the RandomHSV object for random HSV (Hue, Saturation, Value) augmentation.

        This class applies random adjustments to the HSV channels of an image within specified limits.

        Args:
            hgain (float): Maximum variation for hue. Should be in the range [0, 1].
            sgain (float): Maximum variation for saturation. Should be in the range [0, 1].
            vgain (float): Maximum variation for value. Should be in the range [0, 1].
        """
        super().__init__()
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def get_params(self):
        gains = np.random.uniform(-1, 1, 3) * np.array([self.hgain, self.sgain, self.vgain]) + 1.0
        return {"gains": gains}

    def apply_to_target(self, target: np.ndarray, category: str, **params) -> np.ndarray:
        if category != "img":
            return target
        assert target.ndim == 3 and target.shape[2] == 3, (
            f"RandomHSV can only be used for RGB image, but got image with shape {target.shape}."
        )

        h_gain, s_gain, v_gain = params["gains"]
        if (self.hgain == 0) and (self.sgain == 0) and (self.vgain == 0):
            return target

        hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Construct LUT
        x = np.arange(256, dtype=np.uint8)

        lut_h = ((x.astype(np.float32) * h_gain) % 180).astype(np.uint8)
        lut_s = np.clip(x.astype(np.float32) * s_gain, 0, 255).astype(np.uint8)
        lut_v = np.clip(x.astype(np.float32) * v_gain, 0, 255).astype(np.uint8)

        h = cv2.LUT(h, lut_h)
        s = cv2.LUT(s, lut_s)
        v = cv2.LUT(v, lut_v)

        hsv_aug = cv2.merge((h, s, v))
        target = cv2.cvtColor(hsv_aug, cv2.COLOR_HSV2BGR)

        return target


class RandomFlip(TransformBase):
    """
    Randomly flip images horizontally or vertically with corresponding adjustments to annotations.

    This augmentation helps improve model robustness to object orientations and viewpoints.

    Attributes:
        p (float): Probability of applying the flip. Must be between 0 and 1.
        direction (str): Direction of flip, either 'horizontal' or .
        flip_idx (array-like): Index mapping for flipping keypoints, if applicable.
    """

    def __init__(
        self,
        p: float = 0.5,
        direction: Literal["horizontal", "vertical"] = "horizontal",
        flip_idx: list[int] | None = None,
    ) -> None:
        """
        Initialize RandomFlip transform.

        Args:
            p (float): The probability of applying the flip. Must be between 0 and 1.
            direction (str): The direction to apply the flip. Must be 'horizontal' or 'vertical'.
            flip_idx (List[int] | None): Index mapping for flipping keypoints, if any.
        """
        assert direction in {"horizontal", "vertical"}, f"Support direction `horizontal` or `vertical`, got {direction}"
        assert 0 <= p <= 1.0, f"The probability should be in range [0, 1], but got {p}."

        super().__init__(p=p)
        self.direction = direction
        self.flip_idx = flip_idx

    def get_params_on_sample(self, sample: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        return {"size0": self.get_target_size(sample)}

    def apply_to_target(self, target: np.ndarray, **params: dict[str, Any]) -> np.ndarray:
        if self.direction == "vertical":
            return np.flipud(target)
        else:  # "horizontal"
            return np.fliplr(target)

    def apply_to_instances(self, instances: Instances, size0: tuple[int, int], **params: dict[str, Any]):
        instances.convert_bbox(format="xywh")
        h0, w0 = size0
        h0 = 1 if instances.normalized else h0
        w0 = 1 if instances.normalized else w0

        if self.direction == "vertical":
            instances.flipud(h0)
        else:
            instances.fliplr(w0)
            # For keypoints
            if self.flip_idx is not None and instances.keypoints is not None:
                instances.keypoints = instances.keypoints[:, self.flip_idx, :]
        return instances

    def apply_to_mask(self, mask: np.ndarray, **params: dict[str, Any]) -> np.ndarray | None:
        if mask.size == 0:
            return mask

        if self.direction == "vertical":
            masks_flipped = np.flip(mask, axis=-2)  # flip H
        else:
            masks_flipped = np.flip(mask, axis=-1)  # flip W

        return masks_flipped  # (N, newH, newW)


class LetterBox(TransformBase):
    """Resize image and padding for detection, instance segmentation, pose.

    This class resizes and pads images to a specified shape while preserving aspect ratio. It also updates corresponding
    labels and bounding boxes.

    Args:
        dsize (tuple[int, int]): Target size (height, width) for the resized image.
        auto (bool): If True, use minimum rectangle to resize. If False, use dsize directly.
        scale_fill (bool): If True, stretch the image to dsize without padding.
        scaleup (bool): If True, allow scaling up. If False, only scale down.
        center (bool): If True, center the placed image. If False, place image in top-left corner.
        stride (int): Stride of the model (e.g., 32 for YOLOv5).
        padding_value (int): Value for padding the image. Default is 114.
        interpolation (int): Interpolation method for resizing. Default is cv2.INTER_LINEAR.
    """

    def __init__(
        self,
        dsize: tuple[int, int] = (640, 640),
        auto: bool = False,
        scale_fill: bool = False,
        scaleup: bool = True,
        center: bool = True,
        stride: int = 32,
        interpolation: int = cv2.INTER_LINEAR,
    ):
        """Initialize LetterBox object for resizing and padding images.

        This class is designed to resize and pad images for object detection, instance segmentation, and pose estimation
        tasks. It supports various resizing modes including auto-sizing, scale-fill, and letterboxing.
        """
        super().__init__()
        self.dsize = dsize
        self.auto = auto
        self.scale_fill = scale_fill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center  # Put the image in the middle or top-left
        self.interpolation = interpolation

    def get_params_on_sample(self, sample: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        sample_params = super().get_params_on_sample(sample, params)
        size0 = sample_params["size0"]

        dsize = sample.pop("rect_shape", self.dsize)
        if isinstance(dsize, int):
            dsize = (dsize, dsize)

        # Scale ratio (new / old)
        r = min(dsize[0] / size0[0], dsize[1] / size0[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = round(size0[1] * r), round(size0[0] * r)
        dw, dh = dsize[1] - new_unpad[0], dsize[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scale_fill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (dsize[1], dsize[0])
            ratio = dsize[1] / size0[1], dsize[0] / size0[0]  # width, height ratios

        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        top, bottom = round(dh - 0.1) if self.center else 0, round(dh + 0.1)
        left, right = round(dw - 0.1) if self.center else 0, round(dw + 0.1)

        sample_params.update(
            {
                "new_unpad": new_unpad,
                "dw": dw,
                "dh": dh,
                "ratio": ratio,
                "top": top,
                "bottom": bottom,
                "left": left,
                "right": right,
                "dsize": dsize,
            }
        )
        return sample_params

    def apply_to_target(
        self,
        target: np.ndarray,
        size0: tuple[int, int],
        new_unpad: tuple[int, int],
        top: float,
        bottom: float,
        left: float,
        right: float,
        category: str,
        **params: dict[str, Any],
    ):
        # data type
        if np.issubdtype(target.dtype, np.floating):
            padding_value = 0.0
        else:
            if category.lower() == "img":
                padding_value = (114,) * target.shape[2] if target.ndim == 3 else 114
            else:
                padding_value = 0

        if size0[::-1] != new_unpad:  # shape0 = (h, w), new_unpad = (w, h)
            target = cv2.resize(target, new_unpad, interpolation=self.interpolation)

        if target.ndim == 2:
            target = target[..., None]

        # padding float->int
        top, bottom, left, right = int(top), int(bottom), int(left), int(right)

        h, w, c = target.shape
        if target.ndim == 3:
            target = cv2.copyMakeBorder(
                target,
                top,
                bottom,
                left,
                right,
                cv2.BORDER_CONSTANT,
                value=padding_value,
            )
        else:  # (h, w)
            pad_target = np.full(
                (h + top + bottom, w + left + right),
                fill_value=padding_value,
                dtype=target.dtype,
            )
            pad_target[top : top + h, left : left + w] = target
            target = pad_target

        return target

    def apply_to_instances(
        self,
        instances: Instances,
        ratio: tuple[float, float],
        left: float,
        top: float,
        size0: tuple[int, int],
        **params,
    ):
        instances.convert_bbox(format="xyxy")
        instances.denormalize(*size0[::-1])
        instances.scale(*ratio)
        instances.add_padding(left, top)
        return instances

    def apply_to_mask(
        self,
        mask: np.ndarray,
        size0: tuple[int, int],
        new_unpad: tuple[int, int],
        top: float,
        bottom: float,
        left: float,
        right: float,
        **params: dict[str, Any],
    ):
        """
        Resize & pad masks. Expect masks shape: (H, W).
        """
        if mask.size == 0:
            return mask

        top, bottom, left, right = int(top), int(bottom), int(left), int(right)

        if (size0[1], size0[0]) != new_unpad:  # (w, h) vs (w, h)
            mask = cv2.resize(mask, new_unpad, interpolation=cv2.INTER_NEAREST)

        H, W = mask.shape[:2]

        padded = np.zeros((H + top + bottom, W + left + right), dtype=mask.dtype)
        padded[top : top + H, left : left + W] = mask

        return padded

    def update_sample(
        self, sample: dict[str, Any], dsize: tuple[int, int], left: float, top: float, **params
    ) -> dict[str, Any]:
        """Update labels after applying letterboxing to an image.

        This method modifies the bounding box coordinates of instances in the labels to account for resizing and padding
        applied during letterboxing.

        """
        if sample.get("ratio_pad"):
            sample["ratio_pad"] = (sample["ratio_pad"], (left, top))  # for evaluation

        sample["resized_shape"] = dsize

        return sample


class Albumentations(TransformBase):
    """
    Albumentations transformations for image augmentation.

    This class applies various image transformations using the Albumentations library. It includes operations such as
    Blur, Median Blur, conversion to grayscale, Contrast Limited Adaptive Histogram Equalization (CLAHE), random changes
    in brightness and contrast, RandomGamma, and image quality reduction through compression.
    """

    def __init__(
        self,
        p: float = 1.0,
        color_transforms: list[A.BasicTransform] | None = None,
        spatial_transforms: list[A.BasicTransform] | None = None,
    ):
        """
        Initialize the Albumentations transform object for YOLO bbox formatted parameters.

        This class applies various image augmentations using the Albumentations library, including Blur, Median Blur,
        conversion to grayscale, Contrast Limited Adaptive Histogram Equalization, random changes of brightness and
        contrast, RandomGamma, and image quality reduction through compression.

        Args:
            p (float): Probability of applying the augmentations. Must be between 0 and 1.
            color_transforms (list[A.BasicTransform], optional): A list of classes in Albumentations used for RGB color
            transformations.
            spatial_transforms (list[A.BasicTransform], optional): A list of classes in Albumentations used for sprtial
            transformations.
        """
        super().__init__(p=p)

        self.color_compose = None
        self.spatial_compose = None

        # color transforms, only to RGB
        if color_transforms is None:
            self.color_transforms = [
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
                A.RandomBrightnessContrast(p=0.0),
                A.RandomGamma(p=0.0),
                A.ImageCompression(quality_range=(75, 100), p=0.5),
            ]
        else:
            self.color_transforms = color_transforms

        # spatial_transforms
        self.spatial_transforms = spatial_transforms or []

    def get_params_on_sample(self, sample: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        sample_params = {"size0": self.get_target_size(sample)}

        # additional targets for spatial transform
        additional_targets = {}
        if "ir" in sample:
            additional_targets["ir"] = "image"
        if "depth" in sample:
            additional_targets["depth"] = "image"
        sample_params["additional_targets"] = additional_targets

        # mask & masks
        has_mask = "mask" in sample  # np.ndarray
        has_masks = "masks" in sample  # np.ndarray or np.empty
        sample_params["has_mask"] = has_mask
        sample_params["has_masks"] = has_masks

        # bboxes / segments / keypoints
        compose_kwargs = {}
        has_instances = ("instances" in sample) and (len(sample["instances"]) > 0)  # bboxes is not empty
        sample_params["has_instances"] = has_instances
        if has_instances:
            assert "cls" in sample and len(sample["cls"]) == len(sample["instances"]), (
                "'instances' and 'cls' must appear in the sample simultaneously and have same length for Yolo task."
            )
            compose_kwargs["bbox_params"] = A.BboxParams(format="pascal_voc", label_fields=["class_labels"])
            has_segments = sample["instances"].segments is not None and len(sample["instances"].segments) > 0
            if has_segments or sample["instances"].keypoints is not None:
                compose_kwargs["keypoint_params"] = A.KeypointParams(format="xy", remove_invisible=False)

        # create transforms
        if self.color_transforms and len(self.color_transforms) > 0:  # color transform
            self.color_compose = A.Compose(self.color_transforms)
            if hasattr(self.color_compose, "set_random_seed"):
                self.color_compose.set_random_seed(random.randint(0, 2**31 - 1))

        if self.spatial_transforms and len(self.spatial_transforms) > 0:  # spatial transform
            self.spatial_compose = A.Compose(
                self.spatial_transforms, additional_targets=additional_targets, **compose_kwargs
            )
            if hasattr(self.spatial_compose, "set_random_seed"):
                self.spatial_compose.set_random_seed(random.randint(0, 2**31 - 1))

        return sample_params

    def apply_with_params(self, sample: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        # ----------- color transform -----------
        if "img" in sample and self.color_compose is not None:
            img = sample["img"]
            if img is not None and img.ndim == 3 and img.shape[2] == 3:
                sample["img"] = self.color_compose(image=img)["image"]

        # ---------- spatial transform ----------
        if self.spatial_compose is None:
            return sample

        h0, w0 = params["size0"]
        inputs = {}

        if "img" in sample:  # image is the main img in Albumentations
            inputs["image"] = sample["img"]
        else:
            inputs["image"] = np.zeros((h0, w0, 3), dtype=np.uint8)  # pseudo image
        if "ir" in sample:
            inputs["ir"] = sample["ir"][..., None]  # Albumentations requires images to be 3-dimensional.
        if "depth" in sample:
            inputs["depth"] = sample["depth"][..., None]  # Albumentations requires images to be 3-dimensional.

        # bboxes & cls
        has_instances = params.get("has_instances", False)
        if has_instances:
            cls: np.ndarray = sample["cls"]  # not empty
            instances: Instances = sample["instances"]
            instances.convert_bbox("xyxy")
            instances.denormalize(w0, h0)  # [xmin, ymin, xmax, ymax]
            inputs["bboxes"] = instances.bboxes
            inputs["class_labels"] = cls.squeeze()

            # segments (N, 1000, 2)
            has_segments = sample["instances"].segments is not None and len(instances.segments) > 0
            if has_segments:
                segs = deepcopy(instances.segments).reshape(-1, 2)
                inputs["keypoints"] = segs

            # keypoints (N, nkpt, 3) (share keypoints pipleline)
            has_keypoints = instances.keypoints is not None
            if has_keypoints:
                kpts = deepcopy(instances.keypoints).reshape(-1, 3)
                vis = kpts[..., [2]]
                kpts = kpts[..., :2]
                inputs["keypoints"] = kpts

        # mask
        if params["has_mask"]:
            inputs["mask"] = sample["mask"]  # semantic (H, W)

        # masks
        if params["has_masks"]:
            inputs["masks"] = sample["masks"]  # instances (N, H, W) or (0, H, W)

        out = self.spatial_compose(**inputs)

        if "img" in sample:
            sample["img"] = out["image"]
        if "ir" in sample:
            sample["ir"] = out["ir"].squeeze()
        if "depth" in sample:
            sample["depth"] = out["depth"].squeeze()

        h, w = next(iter([sample[key].shape[:2] for key in self._targets if key in sample]))

        if has_instances:
            new_segs = sample["instances"].segments  # origin segments data
            new_kpts = sample["instances"].keypoints  # origin keypoints data
            if "bboxes" in out:
                new_bboxes = out["bboxes"]
                new_cls = out["class_labels"].reshape(-1, 1)
                sample["cls"] = new_cls

            if "keypoints" in out:
                kps_out = out["keypoints"]

                if has_segments:
                    n, _ = new_segs.shape[:2]  # N, M
                    segs = kps_out.reshape(n, -1, 2)
                    segs[..., 0] = segs[..., 0].clip(0, w)
                    segs[..., 1] = segs[..., 1].clip(0, h)
                    new_segs = segs

                elif has_keypoints:
                    n = len(new_kpts)
                    kpts = kps_out
                    out_mask = (new_kpts[:, 0] < 0) | (new_kpts[:, 1] < 0) | (new_kpts[:, 0] > w) | (new_kpts[:, 1] > h)
                    vis[out_mask] = 0
                    new_kpts = np.concatenate([new_kpts, vis], axis=-1).reshape(n, -1, 3)

            new_instances = Instances(
                bboxes=new_bboxes, segments=new_segs, keypoints=new_kpts, bbox_format="xyxy", normalized=False
            )
            new_instances.clip(w, h)
            sample["instances"] = new_instances

        if "mask" in out:
            sample["mask"] = out["mask"]

        if "masks" in out:
            masks: np.ndarray = sample["masks"]
            if masks.size == 0:  # empty
                sample["masks"] = np.zeros((0, h, w), masks.dtype)
            else:
                sample["masks"] = out["masks"]

        return sample


class Format(TransformBase):
    """
    A class for formatting image annotations for object detection, instance segmentation, and pose estimation tasks.

    This class standardizes image and instance annotations to be used by the `collate_fn` in PyTorch DataLoader.

    Attributes:
        bbox_format (str): Format for bounding boxes. Options are 'xywh' or 'xyxy'.
        normalize (bool): Whether to normalize bounding boxes.
        return_mask (bool): Whether to return instance masks for segmentation.
        return_keypoint (bool): Whether to return keypoints for pose estimation.
        return_obb (bool): Whether to return oriented bounding boxes.
        mask_ratio (int): Downsample ratio for masks.
        mask_overlap (bool): Whether to overlap masks.
        batch_idx (bool): Whether to keep batch indexes.
        bgr (float): The probability to return BGR images.
    """

    def __init__(
        self,
        bbox_format: str = "xywh",
        normalize: bool = True,
        return_mask: bool = False,
        mask_mode: Literal["semantic", "instance"] | None = None,
        mask_overlap: bool | None = None,
        return_keypoint: bool = False,
        kpt_shape: tuple[int, int] | None = None,
        return_obb: bool = False,
        mask_ratio: int = 4,
        batch_idx: bool = True,
        bgr: float = 0.0,
    ):
        """
        Initializes the Format class with given parameters for image and instance annotation formatting.

        This class standardizes image and instance annotations for object detection, instance segmentation, and pose
        estimation tasks, preparing them for use in PyTorch DataLoader's `collate_fn`.

        Args:
            bbox_format (str): Format for bounding boxes. Options are 'xywh', 'xyxy', etc.
            normalize (bool): Whether to normalize bounding boxes to [0,1].
            return_mask (bool): If True, returns instance masks for segmentation tasks.
            return_keypoint (bool): If True, returns keypoints for pose estimation tasks.
            return_obb (bool): If True, returns oriented bounding boxes.
            mask_ratio (int): Downsample ratio for masks.
            mask_overlap (bool): If True, allows mask overlap.
            batch_idx (bool): If True, keeps batch indexes.
            bgr (float): Probability of returning BGR images instead of RGB.
        """
        super().__init__()
        self.bbox_format = bbox_format
        self.normalize = normalize
        self.return_mask = return_mask  # set False when training detection only
        self.mask_mode = mask_mode
        self.return_keypoint = return_keypoint
        self.return_obb = return_obb
        self.mask_ratio = mask_ratio
        self.mask_overlap = mask_overlap
        self.batch_idx = batch_idx  # keep the batch indexes
        self.bgr = bgr
        self.kpt_shape = kpt_shape

    def get_params_on_sample(self, sample: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        return {"size0": self.get_target_size(sample), "original_sample": sample, "update_sample": {}}

    def apply_to_target(self, target: np.ndarray, **params: dict[str, Any]) -> torch.Tensor:
        """
        Formats a target image from a Numpy array to a PyTorch tensor.
        """
        if len(target.shape) < 3:
            target = np.expand_dims(target, -1)
        target = target.transpose(2, 0, 1)
        target = np.ascontiguousarray(target[::-1] if random.uniform(0, 1) > self.bgr else target)
        target = torch.from_numpy(target.copy())
        return target

    def apply_to_instances(self, instances: Instances, size0: tuple[int, int], **params: dict[str, Any]) -> Instances:
        """
        Formats instance annotations for object detection, instance segmentation, and pose estimation tasks.
        """
        h, w = size0
        original_sample = params["original_sample"]
        update_sample = params["update_sample"]
        cls = original_sample["cls"]
        instances.convert_bbox(format=self.bbox_format)
        instances.denormalize(w, h)

        nl = len(instances)  # bboxes/segments number
        if self.return_mask:
            # return semantic mask
            if self.mask_mode == "semantic" and "mask" not in original_sample:
                mask, instances, cls = self._segments2masks(instances, cls, w, h)
                original_sample["cls"] = cls
                update_sample["mask"] = torch.from_numpy(mask)

            # return instance masks
            if self.mask_mode == "instance" and "masks" not in original_sample:
                if nl:
                    masks, instances, cls = self._segments2masks(instances, cls, w, h)
                    original_sample["cls"] = cls
                else:
                    masks = torch.zeros(1 if self.mask_overlap else nl, h // self.mask_ratio, w // self.mask_ratio)
                update_sample["masks"] = torch.from_numpy(masks)

            elif self.mask_mode == "instance" and "masks" in original_sample:
                masks = original_sample["masks"]
                if self.mask_overlap:
                    if masks.ndim != 3:  # multi_mask
                        raise ValueError(f"The dimension of masks must be 3, but got {masks.ndim}.")
                    masks, sorted_idx = masks_to_overlap(masks)
                    original_sample["cls"] = cls[sorted_idx]
                    instances = instances[sorted_idx]
                    update_sample["masks"] = torch.from_numpy(masks)

        return instances

    def _segments2masks(
        self, instances: Instances, cls: np.ndarray, w: int, h: int
    ) -> tuple[np.ndarray, Instances, np.ndarray]:
        """
        Converts polygon segments to bitmap masks.
        """
        segments = instances.segments

        if self.mask_mode == "semantic":
            # Convert to semantic masks
            seg_masks = polygons2masks(
                (h, w), segments, color=1, downsample_ratio=1
            )  # segments [N, 1000, 2], seg_masks [N, H, W]
            semantic_mask = np.zeros((h, w), dtype=np.uint8)
            for i in range(len(segments)):
                semantic_mask[seg_masks[i] == 1] = cls[i] + 1  # background=0
            if self.mask_ratio > 1:
                semantic_mask = cv2.resize(
                    semantic_mask,
                    (w // self.mask_ratio, h // self.mask_ratio),
                    interpolation=cv2.INTER_NEAREST,
                )
            return semantic_mask, instances, cls

        elif self.mask_mode == "instance":
            # Convert to instance masks
            if self.mask_overlap:
                masks, sorted_idx = polygons2masks_overlap((h, w), segments, downsample_ratio=self.mask_ratio)
                masks = masks[None]  # (640, 640) -> (1, 640, 640)
                instances = instances[sorted_idx]
                cls = cls[sorted_idx]
            else:
                masks = polygons2masks((h, w), segments, color=1, downsample_ratio=self.mask_ratio)

            return masks, instances, cls

    def apply_to_mask(self, mask: np.ndarray, **params: dict[str, Any]) -> torch.Tensor:
        return torch.from_numpy(mask)

    def apply_to_masks(self, masks: np.ndarray, **params: dict[str, Any]) -> torch.Tensor:
        return torch.from_numpy(masks)

    def update_sample(self, sample: dict[str, Any], size0: tuple[int, int], **params: dict[str, Any]) -> dict[str, Any]:
        h, w = size0
        sample.update(params["update_sample"])

        if "instances" in sample and "cls" in sample:  # yolo task
            cls = sample.pop("cls")
            sample["cls"] = torch.from_numpy(cls)
            instances = sample.pop("instances")
            nl = len(instances)  # bboxes number
            sample["bboxes"] = torch.from_numpy(instances.bboxes)
            if self.return_keypoint:
                sample["keypoints"] = (
                    torch.from_numpy(instances.keypoints)
                    if instances.keypoints is not None
                    else torch.zeros((0, self.kpt_shape[0], self.kpt_shape[1]))
                )
                if self.normalize:
                    sample["keypoints"][..., 0] /= w
                    sample["keypoints"][..., 1] /= h
            if self.return_obb:
                sample["bboxes"] = (
                    xyxyxyxy2xywhr(torch.from_numpy(instances.segments))
                    if len(instances.segments)
                    else torch.zeros((0, 5))
                )
            # NOTE: need to normalize obb in xywhr format for width-height consistency
            if self.normalize:
                sample["bboxes"][:, [0, 2]] /= w
                sample["bboxes"][:, [1, 3]] /= h

        elif "instances" not in sample and "cls" in sample:
            if "masks" not in sample:  # classification task
                nl = 1
            else:
                nl = len(sample["masks"])
            sample["cls"] = torch.from_numpy(sample["cls"])

        # Then we can use collate_fn
        if self.batch_idx:
            sample["batch_idx"] = torch.zeros(nl, dtype=torch.long)

        if not self.return_mask:
            sample.pop("mask", None)
            sample.pop("masks", None)

        return sample


# Yolov8 augmentations -------------------------------------------------------------------------------------------------
def v8_transforms(dataset, imgsz, hyp, stretch=False):
    """
    Applies a series of image transformations for training.

    This function creates a composition of image augmentation techniques to prepare images for YOLO training.
    It includes operations such as mosaic, copy-paste, random perspective, mixup, and various color adjustments.

    Args:
        dataset (Dataset): The dataset object containing image data and annotations.
        imgsz (int): The target image size for resizing.
        hyp (Namespace): A dictionary of hyperparameters controlling various aspects of the transformations.
        stretch (bool): If True, applies stretching to the image. If False, uses LetterBox resizing.
    """
    mosaic = Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic)
    affine = RandomPerspective(
        degrees=hyp.degrees,
        translate=hyp.translate,
        scale=hyp.scale,
        shear=hyp.shear,
        perspective=hyp.perspective,
        pre_transform=None if stretch else LetterBox(dsize=(imgsz, imgsz)),
    )

    pre_transform = Compose([mosaic, affine])
    if hyp.copy_paste_mode == "flip":
        pre_transform.insert(1, CopyPaste(p=hyp.copy_paste, mode=hyp.copy_paste_mode))
    else:
        pre_transform.append(
            CopyPaste(
                dataset,
                pre_transform=Compose([Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic), affine]),
                p=hyp.copy_paste,
                mode=hyp.copy_paste_mode,
            )
        )
    flip_idx = dataset.hyp.get("flip_idx", [])  # for keypoints augmentation
    if dataset.use_keypoints:
        kpt_shape = dataset.hyp.get("kpt_shape", None)
        if len(flip_idx) == 0 and hyp.fliplr > 0.0:
            hyp.fliplr = 0.0
            LOGGER.warning("WARNING ⚠️ No 'flip_idx' array defined in data.yaml, setting augmentation 'fliplr=0.0'")
        elif flip_idx and (len(flip_idx) != kpt_shape[0]):
            raise ValueError(f"data.yaml flip_idx={flip_idx} length must be equal to kpt_shape[0]={kpt_shape[0]}")

    return Compose(
        [
            pre_transform,
            MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),
            Albumentations(p=1.0),
            RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
            RandomFlip(direction="vertical", p=hyp.flipud),
            RandomFlip(direction="horizontal", p=hyp.fliplr, flip_idx=flip_idx),
        ]
    )  # transforms


# Classification augmentations -----------------------------------------------------------------------------------------
def classify_transforms(
    size=224,
    mean=DEFAULT_MEAN,
    std=DEFAULT_STD,
    interpolation="BILINEAR",
    crop_fraction: float = DEFAULT_CROP_FRACTION,
):
    """
    Creates a composition of image transforms for classification tasks.

    This function generates a sequence of torchvision transforms suitable for preprocessing images
    for classification models during evaluation or inference. The transforms include resizing,
    center cropping, conversion to tensor, and normalization.

    Args:
        size (int | tuple): The target size for the transformed image. If an int, it defines the shortest edge. If a
            tuple, it defines (height, width).
        mean (tuple): Mean values for each RGB channel used in normalization.
        std (tuple): Standard deviation values for each RGB channel used in normalization.
        interpolation (str): Interpolation method of either 'NEAREST', 'BILINEAR' or 'BICUBIC'.
        crop_fraction (float): Fraction of the image to be cropped.

    Returns:
        (torchvision.transforms.Compose): A composition of torchvision transforms.

    Examples:
        >>> transforms = classify_transforms(size=224)
        >>> img = Image.open("path/to/image.jpg")
        >>> transformed_img = transforms(img)
    """
    import torchvision.transforms as T  # scope for faster 'import ultralytics'

    if isinstance(size, (tuple, list)):
        assert len(size) == 2, f"'size' tuples must be length 2, not length {len(size)}"
        scale_size = tuple(math.floor(x / crop_fraction) for x in size)
    else:
        scale_size = math.floor(size / crop_fraction)
        scale_size = (scale_size, scale_size)

    # Aspect ratio is preserved, crops center within image, no borders are added, image is lost
    if scale_size[0] == scale_size[1]:
        # Simple case, use torchvision built-in Resize with the shortest edge mode (scalar size arg)
        tfl = [T.Resize(scale_size[0], interpolation=getattr(T.InterpolationMode, interpolation))]
    else:
        # Resize the shortest edge to matching target dim for non-square target
        tfl = [T.Resize(scale_size)]
    tfl.extend(
        [
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        ]
    )
    return T.Compose(tfl)


# Classification training augmentations --------------------------------------------------------------------------------
def classify_augmentations(
    size=224,
    mean=DEFAULT_MEAN,
    std=DEFAULT_STD,
    scale=None,
    ratio=None,
    hflip=0.5,
    vflip=0.0,
    auto_augment=None,
    hsv_h=0.015,  # image HSV-Hue augmentation (fraction)
    hsv_s=0.4,  # image HSV-Saturation augmentation (fraction)
    hsv_v=0.4,  # image HSV-Value augmentation (fraction)
    force_color_jitter=False,
    erasing=0.0,
    interpolation="BILINEAR",
):
    """
    Creates a composition of image augmentation transforms for classification tasks.

    This function generates a set of image transformations suitable for training classification models. It includes
    options for resizing, flipping, color jittering, auto augmentation, and random erasing.

    Args:
        size (int): Target size for the image after transformations.
        mean (tuple): Mean values for normalization, one per channel.
        std (tuple): Standard deviation values for normalization, one per channel.
        scale (tuple | None): Range of size of the origin size cropped.
        ratio (tuple | None): Range of aspect ratio of the origin aspect ratio cropped.
        hflip (float): Probability of horizontal flip.
        vflip (float): Probability of vertical flip.
        auto_augment (str | None): Auto augmentation policy. Can be 'randaugment', 'augmix', 'autoaugment' or None.
        hsv_h (float): Image HSV-Hue augmentation factor.
        hsv_s (float): Image HSV-Saturation augmentation factor.
        hsv_v (float): Image HSV-Value augmentation factor.
        force_color_jitter (bool): Whether to apply color jitter even if auto augment is enabled.
        erasing (float): Probability of random erasing.
        interpolation (str): Interpolation method of either 'NEAREST', 'BILINEAR' or 'BICUBIC'.

    Returns:
        (torchvision.transforms.Compose): A composition of image augmentation transforms.

    Examples:
        >>> transforms = classify_augmentations(size=224, auto_augment="randaugment")
        >>> augmented_image = transforms(original_image)
    """
    # Transforms to apply if Albumentations not installed
    import torchvision.transforms as T  # scope for faster 'import ultralytics'

    if not isinstance(size, int):
        raise TypeError(f"classify_transforms() size {size} must be integer, not (list, tuple)")
    scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range
    ratio = tuple(ratio or (3.0 / 4.0, 4.0 / 3.0))  # default imagenet ratio range
    interpolation = getattr(T.InterpolationMode, interpolation)
    primary_tfl = [T.RandomResizedCrop(size, scale=scale, ratio=ratio, interpolation=interpolation)]
    if hflip > 0.0:
        primary_tfl.append(T.RandomHorizontalFlip(p=hflip))
    if vflip > 0.0:
        primary_tfl.append(T.RandomVerticalFlip(p=vflip))

    secondary_tfl = []
    disable_color_jitter = False
    if auto_augment:
        assert isinstance(auto_augment, str), f"Provided argument should be string, but got type {type(auto_augment)}"
        # color jitter is typically disabled if AA/RA on,
        # this allows override without breaking old hparm cfgs
        disable_color_jitter = not force_color_jitter

        if auto_augment == "randaugment":
            if TORCHVISION_0_11:
                secondary_tfl.append(T.RandAugment(interpolation=interpolation))
            else:
                LOGGER.warning('"auto_augment=randaugment" requires torchvision >= 0.11.0. Disabling it.')

        elif auto_augment == "augmix":
            if TORCHVISION_0_13:
                secondary_tfl.append(T.AugMix(interpolation=interpolation))
            else:
                LOGGER.warning('"auto_augment=augmix" requires torchvision >= 0.13.0. Disabling it.')

        elif auto_augment == "autoaugment":
            if TORCHVISION_0_10:
                secondary_tfl.append(T.AutoAugment(interpolation=interpolation))
            else:
                LOGGER.warning('"auto_augment=autoaugment" requires torchvision >= 0.10.0. Disabling it.')

        else:
            raise ValueError(
                f'Invalid auto_augment policy: {auto_augment}. Should be one of "randaugment", '
                f'"augmix", "autoaugment" or None'
            )

    if not disable_color_jitter:
        secondary_tfl.append(T.ColorJitter(brightness=hsv_v, contrast=hsv_v, saturation=hsv_s, hue=hsv_h))

    final_tfl = [
        T.ToTensor(),
        T.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        T.RandomErasing(p=erasing, inplace=True),
    ]

    return T.Compose(primary_tfl + secondary_tfl + final_tfl)


# NOTE: keep this class for backward compatibility
class ClassifyLetterBox:
    """
    A class for resizing and padding images for classification tasks.

    This class is designed to be part of a transformation pipeline, e.g., T.Compose([LetterBox(size), ToTensor()]).
    It resizes and pads images to a specified size while maintaining the original aspect ratio.

    Attributes:
        h (int): Target height of the image.
        w (int): Target width of the image.
        auto (bool): If True, automatically calculates the short side using stride.
        stride (int): The stride value, used when 'auto' is True.

    Methods:
        __call__: Applies the letterbox transformation to an input image.

    Examples:
        >>> transform = ClassifyLetterBox(size=(640, 640), auto=False, stride=32)
        >>> img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        >>> result = transform(img)
        >>> print(result.shape)
        (640, 640, 3)
    """

    def __init__(self, size=(640, 640), auto=False, stride=32):
        """
        Initializes the ClassifyLetterBox object for image preprocessing.

        This class is designed to be part of a transformation pipeline for image classification tasks. It resizes and
        pads images to a specified size while maintaining the original aspect ratio.

        Args:
            size (int | Tuple[int, int]): Target size for the letterboxed image. If an int, a square image of
                (size, size) is created. If a tuple, it should be (height, width).
            auto (bool): If True, automatically calculates the short side based on stride. Default is False.
            stride (int): The stride value, used when 'auto' is True. Default is 32.

        Attributes:
            h (int): Target height of the letterboxed image.
            w (int): Target width of the letterboxed image.
            auto (bool): Flag indicating whether to automatically calculate short side.
            stride (int): Stride value for automatic short side calculation.

        Examples:
            >>> transform = ClassifyLetterBox(size=224)
            >>> img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            >>> result = transform(img)
            >>> print(result.shape)
            (224, 224, 3)
        """
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size
        self.auto = auto  # pass max size integer, automatically solve for short side using stride
        self.stride = stride  # used with auto

    def __call__(self, im):
        """
        Resizes and pads an image using the letterbox method.

        This method resizes the input image to fit within the specified dimensions while maintaining its aspect ratio,
        then pads the resized image to match the target size.

        Args:
            im (numpy.ndarray): Input image as a numpy array with shape (H, W, C).

        Returns:
            (numpy.ndarray): Resized and padded image as a numpy array with shape (hs, ws, 3), where hs and ws are
                the target height and width respectively.

        Examples:
            >>> letterbox = ClassifyLetterBox(size=(640, 640))
            >>> image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            >>> resized_image = letterbox(image)
            >>> print(resized_image.shape)
            (640, 640, 3)
        """
        imh, imw = im.shape[:2]
        r = min(self.h / imh, self.w / imw)  # ratio of new/old dimensions
        h, w = round(imh * r), round(imw * r)  # resized image dimensions

        # Calculate padding dimensions
        hs, ws = (math.ceil(x / self.stride) * self.stride for x in (h, w)) if self.auto else (self.h, self.w)
        top, left = round((hs - h) / 2 - 0.1), round((ws - w) / 2 - 0.1)

        # Create padded image
        im_out = np.full((hs, ws, 3), 114, dtype=im.dtype)
        im_out[top : top + h, left : left + w] = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        return im_out


# NOTE: keep this class for backward compatibility
class CenterCrop:
    """
    Applies center cropping to images for classification tasks.

    This class performs center cropping on input images, resizing them to a specified size while maintaining the aspect
    ratio. It is designed to be part of a transformation pipeline, e.g., T.Compose([CenterCrop(size), ToTensor()]).

    Attributes:
        h (int): Target height of the cropped image.
        w (int): Target width of the cropped image.

    Methods:
        __call__: Applies the center crop transformation to an input image.

    Examples:
        >>> transform = CenterCrop(640)
        >>> image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        >>> cropped_image = transform(image)
        >>> print(cropped_image.shape)
        (640, 640, 3)
    """

    def __init__(self, size=640):
        """
        Initializes the CenterCrop object for image preprocessing.

        This class is designed to be part of a transformation pipeline, e.g., T.Compose([CenterCrop(size), ToTensor()]).
        It performs a center crop on input images to a specified size.

        Args:
            size (int | Tuple[int, int]): The desired output size of the crop. If size is an int, a square crop
                (size, size) is made. If size is a sequence like (h, w), it is used as the output size.

        Returns:
            (None): This method initializes the object and does not return anything.

        Examples:
            >>> transform = CenterCrop(224)
            >>> img = np.random.rand(300, 300, 3)
            >>> cropped_img = transform(img)
            >>> print(cropped_img.shape)
            (224, 224, 3)
        """
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size

    def __call__(self, im):
        """
        Applies center cropping to an input image.

        This method resizes and crops the center of the image using a letterbox method. It maintains the aspect
        ratio of the original image while fitting it into the specified dimensions.

        Args:
            im (numpy.ndarray | PIL.Image.Image): The input image as a numpy array of shape (H, W, C) or a
                PIL Image object.

        Returns:
            (numpy.ndarray): The center-cropped and resized image as a numpy array of shape (self.h, self.w, C).

        Examples:
            >>> transform = CenterCrop(size=224)
            >>> image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
            >>> cropped_image = transform(image)
            >>> assert cropped_image.shape == (224, 224, 3)
        """
        if isinstance(im, Image.Image):  # convert from PIL to numpy array if required
            im = np.asarray(im)
        imh, imw = im.shape[:2]
        m = min(imh, imw)  # min dimension
        top, left = (imh - m) // 2, (imw - m) // 2
        return cv2.resize(im[top : top + m, left : left + m], (self.w, self.h), interpolation=cv2.INTER_LINEAR)


# NOTE: keep this class for backward compatibility
class ToTensor:
    """
    Converts an image from a numpy array to a PyTorch tensor.

    This class is designed to be part of a transformation pipeline, e.g., T.Compose([LetterBox(size), ToTensor()]).

    Attributes:
        half (bool): If True, converts the image to half precision (float16).

    Methods:
        __call__: Applies the tensor conversion to an input image.

    Examples:
        >>> transform = ToTensor(half=True)
        >>> img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        >>> tensor_img = transform(img)
        >>> print(tensor_img.shape, tensor_img.dtype)
        torch.Size([3, 640, 640]) torch.float16

    Notes:
        The input image is expected to be in BGR format with shape (H, W, C).
        The output tensor will be in RGB format with shape (C, H, W), normalized to [0, 1].
    """

    def __init__(self, half=False):
        """
        Initializes the ToTensor object for converting images to PyTorch tensors.

        This class is designed to be used as part of a transformation pipeline for image preprocessing in the
        Ultralytics YOLO framework. It converts numpy arrays or PIL Images to PyTorch tensors, with an option
        for half-precision (float16) conversion.

        Args:
            half (bool): If True, converts the tensor to half precision (float16). Default is False.

        Examples:
            >>> transform = ToTensor(half=True)
            >>> img = np.random.rand(640, 640, 3)
            >>> tensor_img = transform(img)
            >>> print(tensor_img.dtype)
            torch.float16
        """
        super().__init__()
        self.half = half

    def __call__(self, im):
        """
        Transforms an image from a numpy array to a PyTorch tensor.

        This method converts the input image from a numpy array to a PyTorch tensor, applying optional
        half-precision conversion and normalization. The image is transposed from HWC to CHW format and
        the color channels are reversed from BGR to RGB.

        Args:
            im (numpy.ndarray): Input image as a numpy array with shape (H, W, C) in BGR order.

        Returns:
            (torch.Tensor): The transformed image as a PyTorch tensor in float32 or float16, normalized
                to [0, 1] with shape (C, H, W) in RGB order.

        Examples:
            >>> transform = ToTensor(half=True)
            >>> img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            >>> tensor_img = transform(img)
            >>> print(tensor_img.shape, tensor_img.dtype)
            torch.Size([3, 640, 640]) torch.float16
        """
        im = np.ascontiguousarray(im.transpose((2, 0, 1))[::-1])  # HWC to CHW -> BGR to RGB -> contiguous
        im = torch.from_numpy(im)  # to torch
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0-255 to 0.0-1.0
        return im
