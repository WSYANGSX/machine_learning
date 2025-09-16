from machine_learning.utils import load_cfg, flatten_dict, print_dict

a = {"a": {1: 1, 2: 2}}
a["a"].update({4:4})
print(a)