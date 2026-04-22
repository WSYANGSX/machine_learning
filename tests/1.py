import cv2
import numpy as np
from PIL import Image


img = Image.open("/home/yangxf/WorkSpace/datasets/..datasets/car/masks/train/0cdf5b5d0ce1_01.jpg")
print(img.mode)
img = np.array(img)
print(img.shape)
print(img.max(), img.min())


img = cv2.imread("/home/yangxf/WorkSpace/datasets/..datasets/car/masks/train/0cdf5b5d0ce1_01.jpg", cv2.IMREAD_UNCHANGED)
print(img.shape)
print(img.max(), img.min())
