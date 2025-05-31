from PIL import Image
import albumentations as A
import numpy as np

image = np.array(Image.open("./1.jpg").convert("RGB"), dtype=np.uint8)
Image.fromarray(image).show()

# baseline_pipeline = A.Compose([A.RandomCrop(height=224, width=224, p=1.0), A.HorizontalFlip(p=0.5)])
# result = baseline_pipeline(image=image)
# Image.fromarray(result["image"]).show()
# print(f"Original: {image.shape}, Augmented: {result['image'].shape}")

# # 添加正则化（增强泛化能力）
# enhanced_pipeline = A.Compose(
#     [
#         A.RandomCrop(height=224, width=224, p=1.0),
#         A.HorizontalFlip(p=0.5),
#         # Regularization transforms
#         A.CoarseDropout(
#             num_holes_range=(1, 8), hole_height_range=(10, 32), hole_width_range=(10, 32), fill_value=0, p=0.5
#         ),
#         A.Affine(
#             scale=(0.8, 1.2),  # Conservative scaling
#             rotate=(-15, 15),  # Small rotations
#             p=0.7,
#         ),
#     ]
# )
# result = enhanced_pipeline(image=image)
# Image.fromarray(result["image"]).show()
# print(f"Original: {image.shape}, Augmented: {result['image'].shape}")

# Domain-Specific additions
domain_pipeline = A.Compose(
    [
        A.RandomCrop(height=1080, width=1920, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.CoarseDropout(
            num_holes_range=(1, 8), hole_height_range=(10, 32), hole_width_range=(10, 32), fill_value=0, p=0.5
        ),
        A.Affine(scale=(0.8, 1.2), rotate=(-15, 15), p=0.7),
        # Color robustness
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
        # Optional: noise/blur for robustness
        A.OneOf(
            [
                A.GaussianBlur(blur_limit=(1, 3), p=1.0),
                A.GaussNoise(var_limit=(10, 50), p=1.0),
            ],
            p=0.3,
        ),
    ]
)

result = domain_pipeline(image=image)
Image.fromarray(result["image"]).show()
print(f"Original: {image.shape}, Augmented: {result['image'].shape}")
