import os
import yaml


def print_dict(input_dict: dict, indent: int = 0) -> None:
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
            print("\t", val)


def load_config_from_path(config: str) -> dict:
    assert os.path.splitext(config)[1] == ".yaml" or os.path.splitext(config)[1] == ".yml", (
        "Please ultilize a yaml configuration file."
    )
    with open(config, "r") as f:
        config = yaml.safe_load(f)

    return config


