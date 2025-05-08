import cv2
import fiftyone.zoo as foz


# 加载验证集
dataset = foz.load_zoo_dataset("coco-2017")

img_path = dataset["681c878b435374fa72b56681"].filepath
img = cv2.imread(img_path)
if img is None:
    print(f"无法读取图像：{img_path}")
else:
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
