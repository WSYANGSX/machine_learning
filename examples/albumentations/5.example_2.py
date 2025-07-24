from PIL import Image
from machine_learning.utils.aug_cfg import DEFAULT_YOLOMM_AUG
from machine_learning.utils.detection import yolo2voc
from machine_learning.utils.draw import visualize_img_with_bboxes


from machine_learning.utils.transforms import ImgTransform
from machine_learning.utils.detection import yolo2voc
from machine_learning.utils.draw import visualize_img_with_bboxes


if __name__ == "__main__":
    import numpy as np

    np.set_printoptions(threshold=np.inf)

    img_path = "/home/yangxf/Downloads/align/JPEGImages/FLIR_00002_RGB.jpg"
    image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)

    thermal_path = "/home/yangxf/Downloads/align/JPEGImages/FLIR_00002_PreviewData.jpeg"
    thermal = np.array(Image.open(img_path).convert("L"), dtype=np.uint8)

    labels = np.loadtxt("/home/yangxf/Downloads/align/Annotations/FLIR_00002.txt").reshape(-1, 5)
    bboxes = labels[:, 1:5]

    category_ids = labels[:, 0]
    category_id_to_name = {x: str(x) for x in category_ids}
    visualize_img_with_bboxes(image, yolo2voc(image, bboxes), category_ids, category_id_to_name)
    visualize_img_with_bboxes(thermal, yolo2voc(thermal, bboxes), category_ids, category_id_to_name, "gray")

    transform = ImgTransform(DEFAULT_YOLOMM_AUG, normalize=True, to_tensor=False)

    transformed = transform(
        {"image": image, "thermal": thermal, "bboxes": bboxes, "category_ids": category_ids}, augment=True
    )
    transformed_img = transformed["image"]
    transformed_thermal = transformed["thermal"] * 255
    transformed_bboxes = transformed["bboxes"]
    transformed_category_ids = transformed["category_ids"]

    visualize_img_with_bboxes(
        transformed_img, yolo2voc(transformed_img, transformed_bboxes), transformed["category_ids"], category_id_to_name
    )
    visualize_img_with_bboxes(
        transformed_thermal,
        yolo2voc(transformed_img, transformed_bboxes),
        transformed["category_ids"],
        category_id_to_name,
        "gray",
    )
