import cv2
import numpy as np
from ultralytics.data.utils import polygon2mask
from machine_learning.utils.img import plot_imgs
from ultralytics.utils.ops import resample_segments

img = cv2.imread("/home/yangxf/WorkSpace/machine_learning/tests/augment_test/FLIR_00233_RGB.jpg", cv2.IMREAD_COLOR_RGB)
imgsz = img.shape[:2]
print(imgsz)
h, w = imgsz

lb_file = "/home/yangxf/WorkSpace/machine_learning/tests/augment_test/00233.txt"
with open(lb_file) as f:
    lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
    if any(len(x) > 6 for x in lb):  # is segment
        classes = np.array([x[0] for x in lb], dtype=np.float32)
        segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1, xy2, ...)
segments = np.stack(resample_segments(segments, n=1000), axis=0)  # [N,1000,2]
segments[..., 0] *= w
segments[..., 1] *= h

segments = segments.reshape(-1, 2000)
np.save("segments", segments)

mask = polygon2mask(imgsz, segments, 255)
cv2.imwrite("mask.jpg", mask)


# 比较两个mask
plot_imgs([img, mask], titles=["Original", "polygon2mask"])
