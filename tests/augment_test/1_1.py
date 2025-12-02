import cv2
import torch
import numpy as np
import albumentations as A

from PIL import Image

from machine_learning.utils.plots import plot_imgs, img_tensor2np
from machine_learning.utils.detection import visualize_img_bboxes
from ultralytics.utils.instance import Instances
from ultralytics.data.utils import polygon2mask
from ultralytics.data.augment import (
    RandomPerspective,
    RandomHSV,
    RandomFlip,
    LetterBox,
    Albumentations,
    Format,
)

img = Image.open("/home/yangxf/WorkSpace/machine_learning/tests/augment_test/FLIR_00233_RGB.jpg")
img = np.array(img)

ir = Image.open("/home/yangxf/WorkSpace/machine_learning/tests/augment_test/FLIR_00233_PreviewData.jpeg")
ir = np.array(ir)

mask = Image.open("/home/yangxf/WorkSpace/machine_learning/tests/augment_test/mask.jpg")
mask = np.array(mask)

annotations = np.loadtxt("/home/yangxf/WorkSpace/machine_learning/tests/augment_test/FLIR_00233.txt")
cls = annotations[:, [0]]
bboxes = annotations[:, 1:]

segments = np.load("/home/yangxf/WorkSpace/machine_learning/tests/augment_test/segments.npy")
segments = segments.reshape(-1, 1000, 2)

origin_mask = np.zeros(img.shape[:2], dtype=np.uint8)
for segment in segments:
    # 将点转换为整数坐标
    pts = segment.reshape(-1, 2).astype(np.int32)
    cv2.fillPoly(origin_mask, [pts], color=255)

segments[..., 0] /= img.shape[1]
segments[..., 1] /= img.shape[0]

# empty seg
empty_seg = np.zeros((0, 1000, 2))

instances = Instances(bboxes=bboxes, bbox_format="xywh", normalized=True, segments=segments)
sample = {"img": img, "cls": cls, "instances": instances}

letter_box = LetterBox((640, 640))
sample = letter_box(sample)

random_perspective = RandomPerspective(border=(-512 // 2, -640 // 2))
sample = random_perspective(sample)

# random_hsv = RandomHSV()
# sample = random_hsv(sample)

# random_flip = RandomFlip(p=1, direction="vertical")
# sample = random_flip(sample)


# albumentations = Albumentations(
#     spatial_transforms=[
#         A.Affine(p=1, scale=0.8, shear=10, translate_percent=0.1, rotate=15),
#     ]
# )
# sample = albumentations(sample)

format = Format()
sample = format(sample)

# img = sample["img"]
# ir = sample["ir"]
# cls = sample["cls"]
# instance: Instances = sample["instances"]
# instance.convert_bbox("xyxy")
# instance.denormalize(img.shape[1], img.shape[0])
# bboxes = instance.bboxes
# segments = instance.segments

# mask2 = np.zeros(img.shape[:2], dtype=np.uint8)
# for segment in segments:
#     # 将点转换为整数坐标
#     pts = segment.reshape(-1, 2).astype(np.int32)
#     cv2.fillPoly(mask2, [pts], color=255)

# plot_imgs([sample["img"], sample["ir"], origin_mask, mask2])

# # print(sample)
# visualize_img_bboxes(
#     sample["img"],
#     bboxes,
#     cls.reshape(-1),
#     color=(0, 0, 255),
#     thickness=1,
# )


# format
print("img:", sample["img"])
plot_imgs([img_tensor2np(sample["img"])])
plot_imgs([img_tensor2np(sample["ir"])])
print("ir:", sample["ir"])
print("cls:", sample["cls"])
print("bboxes:", sample["bboxes"])
print("mask:", sample["mask"])
plot_imgs([img_tensor2np(sample["mask"])])

print(img_tensor2np(sample["img"]).shape)
print(sample["bboxes"].numpy().shape)
print(sample["cls"].reshape(-1).numpy().shape)
visualize_img_bboxes(
    img_tensor2np(sample["img"]),
    sample["bboxes"].numpy(),
    sample["cls"].reshape(-1).numpy(),
    color=(0, 0, 255),
    thickness=1,
)
