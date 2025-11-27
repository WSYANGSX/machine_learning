import numpy as np
from machine_learning.utils.detection import visualize_img_bboxes, xywh2xyxy, yolo2voc

targets = np.load("/home/yangxf/WorkSpace/machine_learning/logs/yolo_v13/targets.npy")
imgs = np.load("/home/yangxf/WorkSpace/machine_learning/logs/yolo_v13/inputs.npy")


for i in range(imgs.shape[0]):
    bboxs = targets[..., 2:6][targets[..., 0] == i]
    print(bboxs)
    bboxs = yolo2voc(bboxs, imgs.shape[2], imgs.shape[1])
    print(bboxs)
    cls = targets[..., 1][targets[..., 0] == i]
    visualize_img_bboxes(imgs[i], bboxs, cls)
