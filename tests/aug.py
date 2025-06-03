from machine_learning.utils.augmentation import DEFAULT_AUG, ENHANCED_AUG

if __name__ == "__main__":
    from PIL import Image
    import numpy as np

    img_path = "/home/yangxf/WorkSpace/machine_learning/data/coco-2017/images/train/000000000009.jpg"
    img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    aug = ENHANCED_AUG
    transformed_img = aug(image=img)
    Image.fromarray(transformed_img["image"]).show()
