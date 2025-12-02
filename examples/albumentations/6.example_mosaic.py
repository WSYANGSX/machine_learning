from typing import Any
import numpy as np

from PIL import Image
import albumentations as A

from machine_learning.utils.detection import yolo2voc
from machine_learning.utils.plots import visualize_img_with_bboxes

np.set_printoptions(threshold=np.inf)


def parse_data(id) -> dict[str, Any]:
    img_path = f"./data/Flir_aligned/JPEGImages/FLIR_{id}_RGB.jpg"
    image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)

    thermal_path = f"./data/Flir_aligned/JPEGImages/FLIR_{id}_PreviewData.jpeg"
    thermal = np.array(Image.open(thermal_path).convert("L"), dtype=np.uint8)

    labels = np.loadtxt(f"./data/Flir_aligned/Annotations/FLIR_{id}.txt").reshape(-1, 5)
    bboxes = labels[:, 1:5]
    category_ids = labels[:, 0]

    return {"image": image, "mask": thermal, "bboxes": bboxes, "category_ids": category_ids}


mosaic_transform = A.Compose(
    [
        A.Mosaic(
            grid_yx=(2, 2),
            cell_shape=(320, 320),
            fit_mode="contain",
            target_size=(640, 640),
            metadata_key="mosaic_metadata",
            p=0.6,
        ),
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["category_ids"],
        min_visibility=0.1,
        min_height=0.01,
        min_width=0.01,
        clip=True,
    ),
    p=1,
)

examples = []
ids = ["00002", "00004", "00011", "00099"]
for id in ids:
    examples.append(parse_data(id))

primary_example = examples[0]
primary_data = {
    "image": primary_example["image"],
    "mask": primary_example["mask"],
    "bboxes": primary_example["bboxes"],
    "category_ids": primary_example["category_ids"],
    "mosaic_metadata": examples[1:],
}
result = mosaic_transform(**primary_data)
img = result["image"]
thermal = result["mask"]
print(img.shape)
print(thermal.shape)
bboxes = result["bboxes"]
category_ids = result["category_ids"]

category_id_to_name = {x: str(x) for x in category_ids}

visualize_img_with_bboxes(img, yolo2voc(bboxes, img.shape[1], img.shape[0]), category_ids, category_id_to_name)
visualize_img_with_bboxes(
    thermal, yolo2voc(bboxes, thermal.shape[1], thermal.shape[0]), category_ids, category_id_to_name, "gray"
)
