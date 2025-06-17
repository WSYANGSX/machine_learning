from pathlib import Path
import os
import yaml
from typing import Mapping

a = Path(
    "/home/yangxf/WorkSpace/machine_learning/src/machine_learning/algorithms/detection/yolo_v3/config/yolo_v3.yaml"
)
print(os.path.splitext(a)[1] == ".yaml")
with open(a, "r") as f:
    cfg = yaml.safe_load(f)
print(cfg)
print(isinstance(cfg, Mapping))