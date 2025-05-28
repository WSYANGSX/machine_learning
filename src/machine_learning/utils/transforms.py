import torch
import torch.nn.functional as F
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from .data_utils import xywh2xyxy_np
import torchvision.transforms as transforms


class ImgAug(object):
    def __init__(self, augmentations=None):
        # Albumentations 使用 Compose 替代 iaa.Sequential
        self.augmentations = augmentations if augmentations else A.Compose([])

    def __call__(self, data):
        img, boxes = data
        boxes = np.array(boxes)

        # Convert xywh to xyxy (Albumentations 需要 Pascal VOC 格式 [x_min, y_min, x_max, y_max])
        boxes[:, 1:] = xywh2xyxy_np(boxes[:, 1:])

        # Albumentations 的 bboxes 格式: [[x1, y1, x2, y2, label], ...]
        albumentations_boxes = [np.concatenate([box[1:], [box[0]]]) for box in boxes]

        # 应用增强 (Albumentations 需要明确指定 bbox_params)
        augmented = self.augmentations(
            image=img,
            bboxes=albumentations_boxes,
        )

        img = augmented["image"]
        augmented_boxes = augmented["bboxes"]

        # 转换回 [label, x_center, y_center, w, h] 格式
        new_boxes = np.zeros((len(augmented_boxes), 5))
        for i, box in enumerate(augmented_boxes):
            x1, y1, x2, y2, label = box
            new_boxes[i, 0] = label
            new_boxes[i, 1] = (x1 + x2) / 2  # x_center
            new_boxes[i, 2] = (y1 + y2) / 2  # y_center
            new_boxes[i, 3] = x2 - x1  # width
            new_boxes[i, 4] = y2 - y1  # height

        return img, new_boxes


class RelativeLabels(object):
    def __call__(self, data):
        img, boxes = data
        h, w = img.shape[:2]
        boxes[:, [1, 3]] /= w  # x_center and width
        boxes[:, [2, 4]] /= h  # y_center and height
        return img, boxes


class AbsoluteLabels(object):
    def __call__(self, data):
        img, boxes = data
        h, w = img.shape[:2]
        boxes[:, [1, 3]] *= w  # x_center and width
        boxes[:, [2, 4]] *= h  # y_center and height
        return img, boxes


class PadSquare(object):
    def __call__(self, data):
        img, boxes = data
        h, w = img.shape[:2]

        # 计算需要的 padding
        if h > w:
            pad_left = (h - w) // 2
            pad_right = h - w - pad_left
            padding = ((0, 0), (pad_left, pad_right), (0, 0))
        else:
            pad_top = (w - h) // 2
            pad_bottom = w - h - pad_top
            padding = ((pad_top, pad_bottom), (0, 0), (0, 0))

        # 应用 padding
        padded_img = np.pad(img, padding, mode="constant", constant_values=127.5)

        # 调整 boxes 坐标
        if h > w:
            boxes[:, 1] = (boxes[:, 1] * w + pad_left) / h
            boxes[:, 3] = boxes[:, 3] * w / h
        else:
            boxes[:, 2] = (boxes[:, 2] * h + pad_top) / w
            boxes[:, 4] = boxes[:, 4] * h / w

        return padded_img, boxes


class ToTensor(object):
    def __call__(self, data):
        img, boxes = data
        # 使用 Albumentations 的 ToTensorV2 (会自动归一化到 [0,1] 并转换 C,H,W)
        transform = A.Compose([ToTensorV2()], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

        # 需要将 boxes 转换为 Albumentations 的 yolo 格式 [x_center, y_center, w, h]
        transformed = transform(image=img, bboxes=boxes[:, 1:5], class_labels=boxes[:, 0])

        img_tensor = transformed["image"]
        boxes_tensor = torch.zeros((len(boxes), 6))
        boxes_tensor[:, 1:] = torch.tensor(np.column_stack([transformed["class_labels"], transformed["bboxes"]]))

        return img_tensor, boxes_tensor


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        img, boxes = data
        img = F.interpolate(img.unsqueeze(0), size=self.size, mode="nearest").squeeze(0)
        return img, boxes


# 默认转换 (使用 Albumentations 风格)
DEFAULT_TRANSFORMS = transforms.Compose(
    [
        AbsoluteLabels(),
        PadSquare(),
        RelativeLabels(),
        ToTensor(),
    ]
)
