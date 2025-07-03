import numpy as np
from PIL import Image
import albumentations as A

img = np.array(Image.open("./1.jpg").convert("RGB"), dtype=np.uint8)
Image.fromarray(img).show()

# 水平翻转
flip_h_transform = A.HorizontalFlip(p=0.5)
img1 = flip_h_transform(img)
Image.fromarray(img1["image"]).show()

# # 垂直翻转
# flip_v_transform = A.VerticalFlip(p=0.5)
# img2 = flip_v_transform(image=img)
# Image.fromarray(img2["image"]).show()

# # 亮度调整
# bright_transform = A.RandomBrightnessContrast(p=0.8)
# img3 = bright_transform(image=img)
# Image.fromarray(img3["image"]).show()

# # 模糊调整
# gblur_transform = A.GaussianBlur(p=0.8)
# img4 = gblur_transform(image=img)
# Image.fromarray(img4["image"]).show()

# meblur_transform = A.MedianBlur(p=1.0)
# img5 = meblur_transform(image=img)
# Image.fromarray(img5["image"]).show()

# moblur_transform = A.MotionBlur(p=1.0)
# img6 = moblur_transform(image=img)
# Image.fromarray(img6["image"]).show()

# # resizing/croping
# crop = A.RandomCrop(height=1024, width=1024)
# transformed_img = crop(image=img)
# Image.fromarray(transformed_img["image"]).show()

# resized_crop = A.RandomResizedCrop(size=(1080, 1920))
# transformed_img = resized_crop(image=img)
# Image.fromarray(transformed_img["image"]).show()

# # regularization
# coarse_dropout = A.CoarseDropout()
# transformed_img = coarse_dropout(image=img)
# Image.fromarray(transformed_img["image"]).show()

# # 旋转和放缩
# affine = A.Affine(rotate=(-50, 50), scale=0.5, p=1)
# transformed_img = affine(image=img)
# Image.fromarray(transformed_img["image"]).show()

# # 图像锐化
# sharpening = A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1), p=1)
# sharpened_img = sharpening(image=img)
# Image.fromarray(sharpened_img["image"]).show()

# 移动和放缩
affine = A.Affine(rotate=0, translate_percent=(-0.1, 0.1), scale=(0.8, 1.5), p=1)
transformed_img = affine(image=img)
Image.fromarray(transformed_img["image"]).show()
