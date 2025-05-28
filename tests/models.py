import time
from machine_learning.utils.data_utils import YoloParser

start = time.time()
classes, train_img_paths, val_img_paths, train_labels_paths, val_labels_paths = YoloParser.parse(
    "/home/yangxf/WorkSpace/machine_learning/data/coco-2017"
)
end = time.time()

print(end-start)
