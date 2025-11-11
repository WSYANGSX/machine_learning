import numpy as np
from PIL import Image
from machine_learning.utils.augmentation.img_transforms import RandomPerspective, RandomHSV, RandomFlip, LetterBox
from machine_learning.utils.img import plot_imgs

img = Image.open("/home/yangxf/Downloads/Flir_aligned/JPEGImages/FLIR_00002_RGB.jpg")
img = np.array(img)

ir = Image.open("/home/yangxf/Downloads/Flir_aligned/JPEGImages/FLIR_00002_PreviewData.jpeg")
ir = np.array(ir)

mask = Image.open("/home/yangxf/Downloads/Flir_aligned/Annotations/FLIR_00002_mask.jpg")
mask = np.array(mask)

sample = {"img": img, "ir": ir, "cls": "dog", "masks": mask, "mask_mode": "semantic"}


# random_perspective = RandomPerspective(degrees=45, border=(512 // 2, 640 // 2))
# sample = random_perspective(sample)

# random_hsv = RandomHSV()
# sample = random_hsv(sample)

# random_flip = RandomFlip(p=1, direction="vertical")
# sample = random_flip(sample)

letter_box = LetterBox(dsize=(320, 320))
sample = letter_box(sample)

img = Image.fromarray(sample["img"])


ir = Image.fromarray(sample["ir"])


mask = Image.fromarray(sample["masks"])

plot_imgs([sample["img"], sample["ir"], sample["masks"]])

print(sample)
