import cv2
import albumentations as A
from PIL import Image


if __name__ == "__main__":
    import numpy as np

    np.set_printoptions(threshold=np.inf)

    img_path = "/home/yangxf/WorkSpace/machine_learning/data/coco-2017/images/train/000000000009.jpg"
    image = cv2.imread(img_path, cv2.IMREAD_COLOR_BGR)
    cv2.imshow("1", image)
    cv2.waitKey(0)

#     labels = np.loadtxt("/home/yangxf/Downloads/align/Annotations/FLIR_00002.txt").reshape(-1, 5)
#     bboxes = labels[:, 1:5]
#     category_ids = np.array(labels[:, 0], dtype=np.int32)

# tfs = CustomTransform(
#     augmentation=DEFAULT_YOLO_AUG["transforms"],
#     bbox_params=A.BboxParams(
#         format="yolo",
#         label_fields=[],
#         min_visibility=0.1,
#         min_height=0.01,
#         min_width=0.01,
#         clip=True,
#     ),
#     to_tensor=False,
#     normalize=True,
#     mean=[0, 0, 0],
#     std=[1, 1, 1],
# )

# transformed_data = tfs({"image": image})
# print(transformed_data["image"])
