import os
import json
import glob
import cv2
import numpy as np


def json_to_yolo(json_path, out_dir=None, img_root=None):
    """
    将单个 Labelme JSON 转成 YOLO segmentation txt

    json_path : json 文件路径
    out_dir   : 输出 txt 目录，默认为 json 所在目录
    img_root  : 图片根目录，默认为 json 所在目录
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 确定图片路径和尺寸
    img_name = data.get("imagePath")
    if img_root is None:
        img_root = os.path.dirname(json_path)
    img_path = os.path.join(img_root, img_name)

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {img_path}")
    h, w = img.shape[:2]

    shapes = data.get("shapes", [])
    lines = []

    for s in shapes:
        label = s.get("label", "0")
        # 假设你的 label 就是类别 id，例如 "0" "1" "2"
        try:
            cls_id = int(label)
        except ValueError:
            # 如果不是纯数字，可以在这里自定义映射
            # 例如: cls_map = {"person": 0, "car": 1} 然后 cls_id = cls_map[label]
            cls_id = 0

        points = np.array(s.get("points", []), dtype=float)  # [N,2], 像素坐标
        if points.size == 0:
            continue

        # 像素坐标归一化到 [0,1]
        points[:, 0] /= w  # x / width
        points[:, 1] /= h  # y / height
        points = np.clip(points, 0.0, 1.0)

        # YOLO segmentation 格式: cls x1 y1 x2 y2 ...
        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in points)
        line = f"{cls_id} {coords}"
        lines.append(line)

    if out_dir is None:
        out_dir = os.path.dirname(json_path)
    os.makedirs(out_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(json_path))[0]
    txt_path = os.path.join(out_dir, stem + ".txt")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return txt_path


def batch_json_to_yolo(json_dir, out_dir=None, img_root=None):
    """
    批量转换目录下所有 *.json
    """
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    for jp in json_files:
        txt_path = json_to_yolo(jp, out_dir=out_dir, img_root=img_root)
        print("saved:", txt_path)


if __name__ == "__main__":
    # 单文件示例
    json_path = "/home/yangxf/WorkSpace/machine_learning/tests/augment_test/FLIR_00233_RGB.json"
    json_to_yolo(json_path)

    # 批量示例
    # batch_json_to_yolo("/path/to/json_dir")
