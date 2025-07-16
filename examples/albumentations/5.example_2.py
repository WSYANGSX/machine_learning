from PIL import Image
from machine_learning.utils.aug_cfg import DEFAULT_YOLO_AUG
from machine_learning.utils.detection import yolo2voc
from machine_learning.utils.draw import visualize_img_with_bboxes


from machine_learning.utils.transforms import ImgTransform
from machine_learning.utils.detection import yolo2voc
from machine_learning.utils.draw import visualize_img_with_bboxes


if __name__ == "__main__":
    import numpy as np

    np.set_printoptions(threshold=np.inf)

    img_path = "./data/coco-2017/images/train/000000000025.jpg"
    image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)

    labels = np.loadtxt("./data/coco-2017/labels/train/000000000025.txt").reshape(-1, 5)
    bboxes = labels[:, 1:5]
    bboxes_voc = yolo2voc(image, bboxes)
    category_ids = np.array(labels[:, 0], dtype=np.int32)
    category_id_to_name = {x: str(x) for x in category_ids}
    visualize_img_with_bboxes(image, bboxes_voc, category_ids, category_id_to_name)

    transform = ImgTransform(DEFAULT_YOLO_AUG, normalize=False, to_tensor=True)

    transformed = transform({"image": image, "bboxes": bboxes, "category_ids": category_ids})
    transformed_img = transformed["image"]
    transformed_bboxes = transformed["bboxes"]
    transformed_category_ids = transformed["category_ids"]
    print(transformed_img)
    print(type(transformed_bboxes))
    print(type(transformed_category_ids))

    transformed_bboxes_voc = yolo2voc(transformed_img, transformed_bboxes)
    visualize_img_with_bboxes(transformed_img, transformed_bboxes_voc, transformed["category_ids"], category_id_to_name)
