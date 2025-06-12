import cv2
from PIL import Image
import albumentations as A
import numpy as np
from machine_learning.utils.augmentations import PadShortEdge

# Prepare data with multiple targets
img_path = "./data/coco-2017/images/train/000000102862.jpg"
image = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)

labels = np.loadtxt("./data/coco-2017/labels/train/000000102862.txt").reshape(-1, 5)
bboxes = labels[:, 1:5]

# Spatial transform - affects both image and mask
spatial_pipeline = A.Compose(
    [
        A.RandomCrop(height=224, width=224, p=1.0),
        A.HorizontalFlip(p=0.5),
        # Pixel transform - only affects image
        A.RandomBrightnessContrast(p=0.5),
        PadShortEdge(0, p=1.0),
    ]
)

result = spatial_pipeline(image=image, bboxes=bboxes)

Image.fromarray(result["image"]).show()
print(f"Image shape: {result['image'].shape}")
print(f"Mask shape: {result['bboxes'].shape}")
