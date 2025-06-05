import cv2
from machine_learning.utils.augmentation import DEFAULT_AUG
from machine_learning.utils.draw_utils import visualize_bboxes
from machine_learning.utils.detection import yolo2voc

if __name__ == "__main__":
    import numpy as np

    img_path = "./data/coco-2017/images/train/000000001997.jpg"
    image = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)

    labels = np.loadtxt("./data/coco-2017/labels/train/000000001997.txt").reshape(-1, 5)
    bboxes = labels[:, 1:5]
    bboxes_voc = yolo2voc(image, bboxes)
    category_ids = np.array(labels[:, 0], dtype=np.uint8)
    # We will use the mapping from category_id to the class name
    # to visualize the class label for the bounding box on the image
    category_id_to_name = {0: "dog", 40: "cat", 55: "paiwei"}
    visualize_bboxes(image, bboxes_voc, category_ids, category_id_to_name)

    aug = DEFAULT_AUG
    transformed = aug(image=image, bboxes=bboxes, category_ids=category_ids)
    transformed_img = transformed["image"]
    transformed_bboxes = transformed["bboxes"]
    transformed_category_ids = transformed["category_ids"]
    transformed_bboxes_voc = yolo2voc(transformed_img, transformed_bboxes)
    visualize_bboxes(transformed_img, transformed_bboxes_voc, transformed["category_ids"], category_id_to_name)
