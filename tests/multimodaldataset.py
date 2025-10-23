from machine_learning.utils import load_cfg
from machine_learning.dataset import MMDatasetBase
from machine_learning.dataset.parsers import VedaiParser, FlirAlignedParser, MinistParser
from machine_learning.utils.img import plot_imgs

# data_cfg = load_cfg("/home/yangxf/WorkSpace/machine_learning/src/machine_learning/cfg/datasets/flir_aligned.yaml")
# parser = FlirAlignedParser(data_cfg)

data_cfg = load_cfg("/home/yangxf/WorkSpace/machine_learning/src/machine_learning/cfg/datasets/minist.yaml")
parser = MinistParser(data_cfg)

# data_cfg = load_cfg("/home/yangxf/WorkSpace/machine_learning/src/machine_learning/cfg/datasets/vedai_1024.yaml")
# parser = VedaiParser(data_cfg)

res = parser.parse()
imgs = res["train"]["data"]
irs = res["train"]["data"]
res["train"]["data"] = {"imgs": imgs, "irs": irs}

dataset = MMDatasetBase(data=res["train"]["data"], labels=res["train"]["labels"], cache="ram", fraction=1)
dataset.remove_item(100)
dataset.remove_item(500)
dataset.remove_item(1500)
print(dataset.length)
data, label = dataset.get_data_and_label(59996)
plot_imgs([data["imgs"], data["irs"]])
print(label)
