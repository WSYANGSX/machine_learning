import os
import yaml


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


def load_config_from_path(config: str) -> dict:
    assert os.path.splitext(config)[1] == ".yaml" or os.path.splitext(config)[1] == ".yml", (
        "Please ultilize a yaml configuration file."
    )
    with open(config, "r") as f:
        config = yaml.safe_load(f)

    return config


def print_info_seg() -> None:
    print("=" * 90)


def list_from_txt(file_path: str) -> list[str]:
    lines = []
    with open(file_path, "r") as f:
        for line in f:
            cleaned_line = line.split()[0]
            lines.append(cleaned_line)

    return lines
