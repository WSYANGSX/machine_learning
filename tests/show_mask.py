from PIL import Image
import cv2
import torch
import numpy as np
from machine_learning.utils.ops import img_np2tensor, img_tensor2np
from machine_learning.utils.segment import visualize_mask, generate_gt_edges
from machine_learning.utils.plots import plot_imgs


# img1 = cv2.imread(
#     "/home/yangxf/WorkSpace/datasets/..datasets/VOC2012/SegmentationClassAug/2008_000880.png",
#     cv2.IMREAD_COLOR,
# )

img1 = Image.open("/home/yangxf/WorkSpace/datasets/..datasets/VOC2012/SegmentationClassAug/2008_000880.png")
img1 = np.array(img1)
visualize_mask(img1)
# plot_imgs([img])
# img_tensor = img_np2tensor(img).unsqueeze(0)


img2 = Image.open("/home/yangxf/WorkSpace/datasets/..datasets/VOC2012/SegmentationClassAug/2008_002943.png")
img2 = np.array(img2)


img3 = Image.open("/home/yangxf/WorkSpace/datasets/..datasets/VOC2012/SegmentationClassAug/2008_003819.png")
img3 = np.array(img3)


plot_imgs([img1, img2, img3])

edge1 = generate_gt_edges(torch.tensor(img1), edge_width=1)
edge2 = generate_gt_edges(torch.tensor(img2), edge_width=1)
edge3 = generate_gt_edges(torch.tensor(img3), edge_width=1)

plot_imgs([edge1.squeeze(0).numpy(), edge2.squeeze(0).numpy(), edge3.squeeze(0).numpy()])

# # 显示结果
# cv2.namedWindow("fft shifted (Original Dark)", cv2.WINDOW_NORMAL)
# cv2.imshow("fft shifted (Original Dark)", img_tensor2np(edges.squeeze(0)))

# cv2.waitKey(0)
# cv2.destroyAllWindows()
