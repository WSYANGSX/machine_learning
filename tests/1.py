import numpy as np
from PIL import Image
from machine_learning.utils.augmentation.img_transforms import RandomPerspective, RandomHSV, RandomFlip, LetterBox
from machine_learning.utils.img import plot_imgs
from machine_learning.utils.detection import visualize_img_bboxes
from ultralytics.utils.instance import Instances

img = Image.open("/home/yangxf/Downloads/Flir_aligned/JPEGImages/FLIR_00002_RGB.jpg")
img = np.array(img)

ir = Image.open("/home/yangxf/Downloads/Flir_aligned/JPEGImages/FLIR_00002_PreviewData.jpeg")
ir = np.array(ir)

mask = Image.open("/home/yangxf/Downloads/Flir_aligned/Annotations/FLIR_00002_mask.jpg")
mask = np.array(mask)

annotations = np.loadtxt("/home/yangxf/WorkSpace/dataset/flir_aligned/Annotations/FLIR_00002.txt")
cls = annotations[:, [0]]
bboxes = annotations[:, 1:]

segments = np.zeros((0, 1000, 2), dtype=np.float32)

instance = Instances(bboxes=bboxes, bbox_format="xywh", normalized=True, segments=segments)
sample = {"img": img, "ir": ir, "cls": cls, "masks": mask, "mask_mode": "semantic", "instances": instance}


# random_perspective = RandomPerspective(degrees=90, border=(512 // 2, 640 // 2))
# sample = random_perspective(sample)

# random_hsv = RandomHSV()
# sample = random_hsv(sample)

# random_flip = RandomFlip(p=1, direction="vertical")
# sample = random_flip(sample)

# letter_box = LetterBox(dsize=(320, 320))
# sample = letter_box(sample)

img = sample["img"]
ir = sample["ir"]
mask = sample["masks"]
cls = sample["cls"]
instance: Instances = sample["instances"]
instance.convert_bbox("xyxy")
instance.denormalize(img.shape[1], img.shape[0])
bboxes = instance.bboxes

plot_imgs([sample["img"], sample["ir"], sample["masks"]])

# print(sample)
visualize_img_bboxes(
    sample["img"],
    bboxes,
    cls.reshape(
        -1,
    ),
    color=(0, 0, 255),
    thickness=1,
)
