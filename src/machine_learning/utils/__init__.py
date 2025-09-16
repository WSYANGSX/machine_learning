from typing import Any, Mapping

import os
import yaml
import random

import torch
import numpy as np

from machine_learning.cfg.base import BaseCfg
from machine_learning.types.aliases import FilePath


def set_seed(seed: int = 23) -> None:
    """Set the global random seed variable."""
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Cudnn 设置（确保可重复性，但可能牺牲性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_cfg(cfg: FilePath | Mapping[str, Any]) -> dict:
    if isinstance(cfg, Mapping):
        cfg = dict(cfg)
    else:
        if not (os.path.splitext(cfg)[1] == ".yaml" or os.path.splitext(cfg)[1] == ".yml"):
            raise ValueError("Input path is not a yaml file path.")
        with open(cfg, "r") as f:
            cfg = yaml.safe_load(f)

    return cfg


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
            indent -= 1
        else:
            print(key, ":", end="")
            print(f"\033[{len(key)}D", end="")  # 光标回退，控制对齐
            if isinstance(val, (tuple, list)) and len(val) > 10:
                if isinstance(val, tuple):
                    bracket_open, bracket_close = "(", ")"
                else:
                    bracket_open, bracket_close = "[", "]"
                parts = [str(x) for x in val[:5]] + ["..."] + [str(x) for x in val[-5:]]
                s = bracket_open + ", ".join(parts) + bracket_close
                print("\t" * (7 - indent), s)
            else:
                print("\t" * (7 - indent), val)


def print_cfg(title: str, cfg: dict) -> None:
    """Print cfg dict nicely."""
    print("=" * 110)
    print(f"{title}:")
    print_dict(cfg)
    print("=" * 110)


def cfg2dict(cfg: BaseCfg) -> dict:
    """Convert the cfg class to a dictionary."""
    cfg_dict = {}
    for key, val in cfg.__dict__.items():
        cfg_dict.update({key: val})
    return cfg_dict


def get_gpu_mem() -> float:
    return torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0


def flatten_dict(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Flatten the nested dictionaries into a single-layer dictionary.

    Args:
        dict (dict[str, Any]): The input dictionary.

    Returns:
        dict[str, Any]: The new single-layer dictionary.
    """
    items = []
    for key, value in input_dict.items():
        if isinstance(value, dict):
            items.extend(flatten_dict(value).items())
        else:
            items.append((key, value))

    return dict(items)
