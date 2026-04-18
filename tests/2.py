from PIL import Image
import cv2
import torch
import numpy as np
from machine_learning.utils.ops import img_np2tensor, img_tensor2np
from machine_learning.utils.segment import visualize_mask, generate_gt_edges
from machine_learning.utils.plots import plot_imgs

torch.set_printoptions(threshold=torch.inf)

# img = cv2.imread(
#     "/home/yangxf/WorkSpace/datasets/..datasets/VOC2012/SegmentationClass/2007_000032.png",
#     cv2.IMREAD_UNCHANGED,
# )
img = Image.open("/home/yangxf/WorkSpace/datasets/..datasets/VOC2012/SegmentationObject/2007_000032.png")
img = np.array(img)
plot_imgs([img])
img_tensor = img_np2tensor(img).unsqueeze(0)


# 显示结果
cv2.namedWindow("fft shifted (Original Dark)", cv2.WINDOW_NORMAL)
cv2.imshow("fft shifted (Original Dark)", img_tensor2np(edges.squeeze(0)))

cv2.waitKey(0)
cv2.destroyAllWindows()
