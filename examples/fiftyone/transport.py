import os
import shutil

source_path = "/home/yangxf/WorkSpace/machine_learning/data/coco-2017/images/train/data"
target_path = "/home/yangxf/WorkSpace/machine_learning/data/coco-2017/images/train"

file_names = [f for f in os.listdir(source_path) if os.path.isfile(os.path.join(source_path, f))]
for fn in file_names:
    try:
        shutil.move(os.path.join(source_path, fn), os.path.join(target_path, fn))
        print("文件移动成功")
    except FileNotFoundError:
        print("源文件不存在")
    except PermissionError:
        print("权限不足")
