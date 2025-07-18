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

    img_path = "/home/yangxf/WorkSpace/machine_learning/data/GTOT/BlackCar/v/00001v.png"
    image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)

    thermal_path = "/home/yangxf/WorkSpace/machine_learning/data/GTOT/BlackCar/i/00001i.png"
    thermal = np.array(Image.open(img_path).convert("L"), dtype=np.uint8)

    labels = np.loadtxt("/home/yangxf/WorkSpace/machine_learning/data/GTOT/BlackCar/groundTruth_i.txt").reshape(-1, 4)
    bboxes = labels[[0], :]

    category_ids = np.array([0])
    category_id_to_name = {x: str(x) for x in category_ids}
    visualize_img_with_bboxes(image, bboxes, category_ids, category_id_to_name)
    visualize_img_with_bboxes(thermal, bboxes, category_ids, category_id_to_name, "gray")

    transform = ImgTransform(DEFAULT_YOLOMM_AUG, normalize=True, to_tensor=False)

    transformed = transform(
        {"image": image, "thermal": thermal, "bboxes": bboxes, "category_ids": category_ids}, augment=False
    )
    transformed_img = transformed["image"]
    transformed_thermal = transformed["thermal"] * 255
    transformed_bboxes = transformed["bboxes"]
    transformed_category_ids = transformed["category_ids"]

    visualize_img_with_bboxes(transformed_img, transformed_bboxes, transformed["category_ids"], category_id_to_name)
    visualize_img_with_bboxes(
        transformed_thermal, transformed_bboxes, transformed["category_ids"], category_id_to_name, "gray"
    )
