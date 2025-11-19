import numpy as np

from PIL import Image
from machine_learning.utils.detection import yolo2voc
from machine_learning.utils.transforms import DEFAULT_YOLOMM_AUG
from examples.transforms import ImgTransform
from machine_learning.utils.image import visualize_img_with_bboxes

np.set_printoptions(threshold=np.inf)

if __name__ == "__main__":
    img_path = "./data/Flir_aligned/JPEGImages/FLIR_00099_RGB.jpg"
    image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)

    thermal_path = "./data/Flir_aligned/JPEGImages/FLIR_00099_PreviewData.jpeg"
    thermal = np.array(Image.open(thermal_path).convert("L"), dtype=np.uint8)

    labels = np.loadtxt("./data/Flir_aligned/Annotations/FLIR_00099.txt").reshape(-1, 5)
    bboxes = labels[:, 1:5]

    category_ids = labels[:, 0]
    category_id_to_name = {x: str(x) for x in category_ids}
    visualize_img_with_bboxes(
        image, yolo2voc(bboxes, image.shape[1], image.shape[0]), category_ids, category_id_to_name
    )
    visualize_img_with_bboxes(
        thermal, yolo2voc(bboxes, thermal.shape[1], thermal.shape[0]), category_ids, category_id_to_name, "gray"
    )

    transform = ImgTransform(DEFAULT_YOLOMM_AUG, normalize=True, to_tensor=False)

    transformed = transform(
        {"image": image, "thermal": thermal, "bboxes": bboxes, "category_ids": category_ids}, augment=True
    )
    transformed_img = transformed["image"]
    print(transformed_img.shape)
    transformed_thermal = transformed["thermal"] * 255
    print(transformed_thermal.shape)
    transformed_bboxes = transformed["bboxes"]
    transformed_category_ids = transformed["category_ids"]

    visualize_img_with_bboxes(
        transformed_img,
        yolo2voc(transformed_bboxes, transformed_img.shape[1], transformed_img.shape[0]),
        transformed["category_ids"],
        category_id_to_name,
    )
    visualize_img_with_bboxes(
        transformed_thermal,
        yolo2voc(transformed_bboxes, transformed_thermal.shape[1], transformed_thermal.shape[0]),
        transformed["category_ids"],
        category_id_to_name,
        "gray",
    )
