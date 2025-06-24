import torch
from machine_learning.utils.detection import bbox_iou

a = torch.tensor([[0.2, 0.5, 0.3, 0.4], [0.1, 0.2, 0.35, 0.4]])
b = torch.tensor([[0.2, 0.5, 0.3, 0.4], [0.1, 0.2, 0.35, 0.4], [0.1, 0.2, 0.35, 0.4]])
print(bbox_iou(a, b, bbox_format="coco"))
