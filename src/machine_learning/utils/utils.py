import torch  # noqa F:401


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
