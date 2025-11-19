from PIL import Image
import cv2
import numpy as np
import albumentations as A
from machine_learning.utils.image import plot_imgs

img = Image.open("/home/yangxf/Downloads/Flir_aligned/JPEGImages/FLIR_00002_RGB.jpg")
img = np.array(img)

ir = Image.open("/home/yangxf/Downloads/Flir_aligned/JPEGImages/FLIR_00002_PreviewData.jpeg")
ir = np.array(ir)

additional_targets = {"ir": "image"}
affine = A.Affine(p=1, scale=0.8, shear=10, translate_percent=0.1, rotate=15)
compose = A.Compose(
    [
        A.Affine(p=1, scale=0.8, shear=10, translate_percent=0.1, rotate=15),
        A.CenterCrop(height=280, width=280, p=1),
        A.HorizontalFlip(p=1.0),
        A.Rotate(
            limit=(-15, 15),  # Range
            p=0.7,
        ),
        A.RandomCrop(height=224, width=224, p=1.0),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
    ],
    additional_targets=additional_targets,
)
res = compose(image=img, ir=ir[..., None])
print(res["ir"].shape)
plot_imgs([res["image"], res["ir"].squeeze()])
