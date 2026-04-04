import torch
import cv2
from machine_learning.utils.plots import plot_imgs

mask = cv2.imread(
    "/home/yangxf/WorkSpace/datasets/..datasets/car/masks/train/0cdf5b5d0ce1_01.jpg", cv2.IMREAD_UNCHANGED
)
plot_imgs([mask])
