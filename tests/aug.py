from machine_learning.utils.augmentation import DEFAULT_AUG

if __name__ == "__main__":
    from PIL import Image
    import numpy as np

    img_path = "/home/yangxf/WorkSpace/machine_learning/data/coco-2017/images/train/000000000009.jpg"
    img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    print(img.shape)

    class_bboxes = np.loadtxt(
        "/home/yangxf/WorkSpace/machine_learning/data/coco-2017/labels/train/000000000009.txt"
    ).reshape(-1, 5)
    class_labels = class_bboxes[:, 1]
    bboxes = class_bboxes[:, 1:5]
    print(bboxes)

    aug = DEFAULT_AUG
    transformed_img = aug(image=img, bboxes=bboxes, class_labels=class_labels)
    Image.fromarray(transformed_img["image"]).show()
    print(transformed_img["bboxes"])
