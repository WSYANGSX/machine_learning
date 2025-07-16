import cv2
import albumentations as A
from machine_learning.utils.transforms import CustomTransform
from machine_learning.utils.aug_cfg import DEFAULT_YOLO_AUG


if __name__ == "__main__":
    import numpy as np

    np.set_printoptions(threshold=np.inf)

    img_path = "./data/coco-2017/images/train/000000559214.jpg"
    image = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)

    labels = np.loadtxt("./data/coco-2017/labels/train/000000559214.txt").reshape(-1, 5)
    bboxes = labels[:, 1:5]
    category_ids = np.array(labels[:, 0], dtype=np.int32)

tfs = CustomTransform(
    augmentation=DEFAULT_YOLO_AUG["transforms"],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=[],
        min_visibility=0.1,
        min_height=0.01,
        min_width=0.01,
        clip=True,
    ),
    to_tensor=False,
    normalize=True,
    mean=[0, 0, 0],
    std=[1, 1, 1],
)

transformed_data = tfs({"image": image})
print(transformed_data["image"])
