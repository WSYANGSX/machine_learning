import cv2
import torch
import albumentations as A
from machine_learning.utils.detection import yolo2voc, resize, visualize_img_bboxes
import torchvision.transforms as T

# from machine_learning.utils.augmentations import PadShortEdge
# from machine_learning.utils.detection import yolo2voc
# from machine_learning.utils.draw import visualize_img_with_bboxes
from machine_learning.utils.image import plot_imgs, imgs_np2tensor

if __name__ == "__main__":
    import numpy as np

    np.set_printoptions(threshold=np.inf)

    img_path = "/home/yangxf/WorkSpace/machine_learning/examples/albumentations/1.jpg"
    image = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)
    plot_imgs([image])
    h, w = image.shape[:2]
    print(h, w)
    masks = np.zeros((0, h, w), dtype=np.uint8)

    transform = A.RandomCrop(320, 320)
    res = transform(image=image, masks=masks)
    plot_imgs([res["image"]])
    print(res["image"].shape)
    print(res["masks"].shape)
    # labels = np.loadtxt("./data/coco-2017/labels/train/000000000072.txt").reshape(-1, 5)
    # bboxes = labels[:, 1:5]
    # bboxes_voc = yolo2voc(bboxes, image.shape[1], image.shape[0])
    # category_ids = labels[:, 0]

    # # We will use the mapping from category_id to the class name
    # # to visualize the class label for the bounding box on the image
    # category_id_to_name = {x: str(x) for x in category_ids}
    # visualize_img_with_bboxes(image, bboxes_voc, category_ids, category_id_to_name)

    # # aug = DEFAULT_YOLO_AUG
    # # transformed = aug(image=image, bboxes=bboxes, category_ids=category_ids)
    # # transformed_img = transformed["image"]
    # # transformed_bboxes = transformed["bboxes"]
    # # transformed_category_ids = transformed["category_ids"]
    # # print(type(transformed_category_ids))

    # # transformed_bboxes_voc = yolo2voc(transformed_img, transformed_bboxes)
    # # visualize_img_with_bboxes(transformed_img, transformed_bboxes_voc, transformed["category_ids"], category_id_to_name)

    # aug = A.pytorch.ToTensorV2(transpose_mask=True)
    # transformed = aug(image=image, bboxes=bboxes, category_ids=category_ids)
    # transformed_img = transformed["image"]
    # transformed_bboxes = transformed["bboxes"]
    # transformed_category_ids = transformed["category_ids"]
    # print(transformed_img)
    # print(type(transformed_bboxes))
    # print(type(transformed_category_ids))
    # normalize = T.Normalize(mean=-1, std=1)
    # transformed_img = normalize(transformed_img.float())
    # print(transformed_img)
