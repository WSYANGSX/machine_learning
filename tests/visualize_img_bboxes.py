from PIL import Image
import numpy as np
from machine_learning.utils.detection import visualize_img_bboxes, yolo2voc
from ultralytics.utils.ops import segments2boxes

img = Image.open("/home/yangxf/WorkSpace/datasets/..datasets/coco/images/train2017/000000378077.jpg")
img = np.array(img)
lb_file = "/home/yangxf/WorkSpace/datasets/..datasets/coco/annotations/train2017/000000378077.txt"

# segments to bboxes
with open(lb_file) as f:
    lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
    if any(len(x) > 6 for x in lb):  # is segment
        classes = np.array([x[0] for x in lb], dtype=np.float32)
        segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1, xy2, ...)
        lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
    else:
        lb = np.array(lb, dtype=np.float32)

bboxes = yolo2voc(lb[:, 1:5], img.shape[1], img.shape[0])
cls = lb[:, 0]
visualize_img_bboxes(img, bboxes, cls)
