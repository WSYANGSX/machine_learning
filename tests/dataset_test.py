from PIL import Image
from machine_learning.utils import load_cfg_from_yaml
from machine_learning.dataset import DatasetBase
from machine_learning.dataset.parsers import MinistParser


# data_cfg = load_cfg_from_yaml(
#     "/home/yangxf/WorkSpace/machine_learning/src/machine_learning/cfg/datasets/coco_test.yaml"
# )
# parser = CocoTestParser(data_cfg)
# res = parser.parse()

# dataset = DatasetBase(data=res["train"]["imgs"], labels=res["train"]["labels"], cache="Disk")


data_cfg = load_cfg_from_yaml("/home/yangxf/WorkSpace/machine_learning/src/machine_learning/cfg/datasets/minist.yaml")
parser = MinistParser(data_cfg)
res = parser.parse()

dataset = DatasetBase(data=res["train"]["imgs"], labels=res["train"]["labels"])


Image.fromarray(dataset.data[512]).show()
print(dataset.labels[512])
