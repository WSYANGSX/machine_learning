from PIL import Image
import albumentations as A
import numpy as np

# Prepare data with multiple targets
image = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
mask = np.random.randint(0, 5, (300, 300), dtype=np.uint8)

# Spatial transform - affects both image and mask
spatial_pipeline = A.Compose(
    [
        A.RandomCrop(height=224, width=224, p=1.0),
        A.HorizontalFlip(p=0.5),
        # Pixel transform - only affects image
        A.RandomBrightnessContrast(p=0.5),
    ]
)

result = spatial_pipeline(image=image, mask=mask)

Image.fromarray(result["image"]).show()
Image.fromarray(result["mask"]).show()
print(f"Image shape: {result['image'].shape}")
print(f"Mask shape: {result['mask'].shape}")
print("Spatial alignment maintained between image and mask")
