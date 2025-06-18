import os
import yaml
import random

import torch
import numpy as np

from machine_learning.types.aliases import FilePath


def print_dict(input_dict: dict, indent: int = 0) -> None:
    if not input_dict:
        return

    indent = indent

    for key, val in input_dict.items():
        print("\t" * indent, end="")
        if isinstance(val, dict):
            indent += 1
            print(key, ":")
            print_dict(val, indent)
            indent = 0
        else:
            print(key, ":", end="")
            print(f"\033[{len(key)}D", end="")  # 光标回退，控制对齐
            print("\t" * 5, val)


def load_config_from_yaml(file_path: FilePath) -> dict:
    assert os.path.splitext(file_path)[1] == ".yaml" or os.path.splitext(file_path)[1] == ".yml", (
        "Please ultilize a yaml configuration file."
    )
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def print_segmentation() -> None:
    print("=" * 90)


def list_from_txt(file_path: FilePath) -> list[str]:
    lines = []
    with open(file_path, "r") as f:
        for line in f:
            cleaned_line = line.split()[0]
            lines.append(cleaned_line)

    return lines


def set_seed(seed: int = 23) -> None:
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Cudnn 设置（确保可重复性，但可能牺牲性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
