import os
import yaml
import random

import torch
import numpy as np

from machine_learning.types.aliases import FilePath
from machine_learning.utils.cfg import BaseCfg


def set_seed(seed: int = 23) -> None:
    """Set the global random seed variable."""
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Cudnn 设置（确保可重复性，但可能牺牲性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config_from_yaml(file_path: FilePath) -> dict:
    """Load the yaml file into a dictionary."""
    assert os.path.splitext(file_path)[1] == ".yaml" or os.path.splitext(file_path)[1] == ".yml", (
        "Please ultilize a yaml configuration file."
    )
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def list_from_txt(file_path: FilePath) -> list[str]:
    """Load the lines of txt file into a list."""
    lines = []
    with open(file_path, "r") as f:
        for line in f:
            cleaned_line = line.split()[0]
            lines.append(cleaned_line)

    return lines


def print_dict(input_dict: dict, indent: int = 0) -> None:
    """Print Dictionary nicely."""
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


def print_cfg(title: str, cfg: dict) -> None:
    """Print cfg dict nicely."""
    print("=" * 90)
    print(f"{title}:")
    print_dict(cfg)
    print("=" * 90)


def cfg_to_dict(cfg: BaseCfg) -> dict:
    """Convert the cfg class to a dictionary."""
    cfg_dict = {}
    for key, val in cfg.__dict__.items():
        cfg_dict.update({key: val})
    return cfg_dict