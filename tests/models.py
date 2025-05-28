from machine_learning.utils.data_utils import YoloParser

classes, train_img_paths, val_img_paths, train_labels_paths, val_labels_paths = YoloParser.parse(
    "/home/yangxf/WorkSpace/machine_learning/data/coco-2017"
)

print(classes)
print(train_img_paths)
print(val_img_paths)
print(train_labels_paths)
print(val_labels_paths)
