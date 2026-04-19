from PIL import Image
import cv2
import torch
import numpy as np
from machine_learning.utils.ops import img_np2tensor, img_tensor2np
from machine_learning.utils.segment import visualize_mask, generate_gt_edges
from machine_learning.utils.plots import plot_imgs


# img = cv2.imread(
#     "/home/yangxf/WorkSpace/datasets/..datasets/VOC2012/SegmentationClass/2007_000032.png",
#     cv2.IMREAD_UNCHANGED,
# )
img1 = Image.open("/home/yangxf/WorkSpace/datasets/..datasets/car/masks/test/4b74275babf7_01.jpg")
img1 = np.array(img1)
# plot_imgs([img])
# img_tensor = img_np2tensor(img).unsqueeze(0)


img2 = Image.open("/home/yangxf/WorkSpace/datasets/..datasets/car/masks/test/4baf50a3d8c2_05.jpg")
img2 = np.array(img2)


img3 = Image.open("/home/yangxf/WorkSpace/datasets/..datasets/car/masks/test/4dc8a7fe7c02_12.jpg")
img3 = np.array(img3)


masks = np.stack([img1, img2, img1])

plot_imgs([masks[0], masks[1], masks[2]])

edges = generate_gt_edges(torch.tensor(masks), edge_width=1)

plot_imgs([edges[0].numpy(), edges[1].numpy(), edges[2].numpy()])

# # 显示结果
# cv2.namedWindow("fft shifted (Original Dark)", cv2.WINDOW_NORMAL)
# cv2.imshow("fft shifted (Original Dark)", img_tensor2np(edges.squeeze(0)))

# cv2.waitKey(0)
# cv2.destroyAllWindows()
