import os
import shutil

source_path = "/home/yangxf/WorkSpace/machine_learning/data/coco-2017/images/train"
target_path = "/home/yangxf/WorkSpace/machine_learning/data/coco-2017/labels/validation"

file_names1 = [f for f in os.listdir(source_path) if os.path.isfile(os.path.join(source_path, f))]
file_names2 = [f for f in os.listdir(target_path) if os.path.isfile(os.path.join(target_path, f))]

print(len(file_names1))
print(len(file_names2))