import xml.etree.ElementTree as ET
import os
import argparse
from collections import defaultdict


def parse_xml_annotations(xml_dir):
    """
    解析所有XML文件, 收集类别信息和标注数据

    参数:
        xml_dir (str): XML文件目录路径

    返回:
        tuple: (类别映射字典, 所有标注数据字典)
    """
    class_counter = defaultdict(int)
    all_annotations = {}

    for xml_file in os.listdir(xml_dir):
        if not xml_file.endswith(".xml"):
            continue

        xml_path = os.path.join(xml_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 获取文件名（不含扩展名）
        filename = root.find("filename").text
        base_name = os.path.splitext(filename)[0]

        # 获取图像尺寸
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)

        # 收集所有object
        annotations = []
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            class_counter[class_name] += 1

            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            annotations.append(
                {
                    "class_name": class_name,
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "width": width,
                    "height": height,
                }
            )

        all_annotations[base_name] = annotations

    # 创建类别映射（按字母顺序排序）
    sorted_classes = sorted(class_counter.keys())
    class_map = {cls: idx for idx, cls in enumerate(sorted_classes)}

    return class_map, all_annotations, class_counter


def generate_names_file(class_map, output_dir):
    """
    生成YOLO格式的names文件

    参数:
        class_map (dict): 类别名称到ID的映射
        output_dir (str): 输出目录
    """
    names_path = os.path.join(output_dir, "classes.names")
    with open(names_path, "w") as f:
        for cls in sorted(class_map.keys(), key=lambda x: class_map[x]):
            f.write(f"{cls}\n")
    print(f"类别名称文件已生成: {names_path}")


def generate_yolo_annotations(all_annotations, class_map, output_dir):
    """
    生成YOLO格式的标注文件

    参数:
        all_annotations (dict): 所有标注数据
        class_map (dict): 类别名称到ID的映射
        output_dir (str): 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    for base_name, annotations in all_annotations.items():
        txt_path = os.path.join(output_dir, f"{base_name}.txt")

        with open(txt_path, "w") as f:
            for ann in annotations:
                # 获取类别ID
                class_id = class_map[ann["class_name"]]

                # 计算归一化坐标
                width, height = ann["width"], ann["height"]
                x_center = (ann["xmin"] + ann["xmax"]) / (2.0 * width)
                y_center = (ann["ymin"] + ann["ymax"]) / (2.0 * height)
                bbox_width = (ann["xmax"] - ann["xmin"]) / width
                bbox_height = (ann["ymax"] - ann["ymin"]) / height

                # 写入YOLO格式行
                line = f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n"
                f.write(line)

    print(f"已生成 {len(all_annotations)} 个YOLO标注文件")


def main(xml_dir, output_dir):
    # 解析XML文件
    class_map, all_annotations, class_counter = parse_xml_annotations(xml_dir)

    # 打印类别统计信息
    print("\n检测到的类别统计:")
    for cls, count in sorted(class_counter.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cls}: {count} 个实例")

    print(f"\n总共检测到 {len(class_map)} 个类别")

    # 生成names文件
    generate_names_file(class_map, output_dir)

    # 生成YOLO标注文件
    generate_yolo_annotations(all_annotations, class_map, output_dir)

    # 打印类别映射
    print("\n类别ID映射:")
    for cls, id in class_map.items():
        print(f"  {id}: {cls}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将VOC XML标注转换为YOLO格式并自动生成类别文件")
    parser.add_argument("--xml-dir", required=True, help="包含XML文件的目录路径")
    parser.add_argument("--output-dir", required=True, help="输出YOLO标注文件的目录路径")

    args = parser.parse_args()

    main(args.xml_dir, args.output_dir)
