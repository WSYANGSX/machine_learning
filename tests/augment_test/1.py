import cv2
import numpy as np
import albumentations as A

from PIL import Image
from machine_learning.utils.augmentation.img_transforms import (
    RandomPerspective,
    RandomHSV,
    RandomFlip,
    LetterBox,
    Albumentations,
)
from machine_learning.utils.img import plot_imgs
from machine_learning.utils.detection import visualize_img_bboxes
from ultralytics.utils.instance import Instances
from ultralytics.data.utils import polygon2mask

np.set_printoptions(threshold=np.inf)

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
sample = {"img": img, "ir": ir, "cls": cls, "instances": instances}


random_perspective = RandomPerspective(degrees=45, border=(512 // 2, 640 // 2))
sample = random_perspective(sample)

# random_hsv = RandomHSV()
# sample = random_hsv(sample)

# random_flip = RandomFlip(p=1, direction="vertical")
# sample = random_flip(sample)

# letter_box = LetterBox(dsize=(320, 320))
# sample = letter_box(sample)

# albumentations = Albumentations(
#     spatial_transforms=[
#         A.Affine(p=1, scale=0.8, shear=10, translate_percent=0.1, rotate=15),
#     ]
# )
# sample = albumentations(sample)

img = sample["img"]
ir = sample["ir"]
cls = sample["cls"]
instance: Instances = sample["instances"]
instance.convert_bbox("xyxy")
instance.denormalize(img.shape[1], img.shape[0])
bboxes = instance.bboxes
segments = instance.segments

mask2 = np.zeros(img.shape[:2], dtype=np.uint8)
for segment in segments:
    # 将点转换为整数坐标
    pts = segment.reshape(-1, 2).astype(np.int32)
    cv2.fillPoly(mask2, [pts], color=255)

plot_imgs([sample["img"], sample["ir"], origin_mask, mask2])

# print(sample)
visualize_img_bboxes(
    sample["img"],
    bboxes,
    cls.reshape(-1),
    color=(0, 0, 255),
    thickness=1,
)
