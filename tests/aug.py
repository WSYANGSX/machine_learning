from machine_learning.utils.augmentation import DEFAULT_AUG, ENHANCED_AUG, PadShortEdge

if __name__ == "__main__":
    from PIL import Image
    import numpy as np

    img_path = "/home/yangxf/WorkSpace/machine_learning/examples/albumentations/1.jpg"
    img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    print(img.shape)
    aug = PadShortEdge(0)
    transformed_img = aug(image=img)
    Image.fromarray(transformed_img["image"]).show()
