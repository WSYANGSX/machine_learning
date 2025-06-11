import torch
import cv2
import numpy as np

image = np.random.randint(low=0, high=255, size=(3, 3, 1), dtype=np.uint8)
image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
print(image.shape)
