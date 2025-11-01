from PIL import Image
import cv2
import numpy as np

np.set_printoptions(threshold=np.inf)


img = Image.open("/home/yangxf/WorkSpace/machine_learning/data/flir_aligned/JPEGImages/FLIR_10218_PreviewData.jpeg")
print(img.mode)
img = np.array(img)
print(img.shape)

img = cv2.imread("/home/yangxf/WorkSpace/machine_learning/data/flir_aligned/JPEGImages/FLIR_10218_PreviewData.jpeg")
print(img.shape)
