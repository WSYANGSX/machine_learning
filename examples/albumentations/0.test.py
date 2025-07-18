import albumentations as A

# Create transform instances
flip_transform = A.HorizontalFlip(p=0.5)
brightness_transform = A.RandomBrightnessContrast(p=0.3)
blur_transform = A.GaussianBlur(p=0.2)

# Create pipeline with these specific instances
pipeline = A.Compose([])
print(bool(pipeline))