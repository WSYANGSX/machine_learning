from machine_learning.utils import load_cfg
from machine_learning.dataset import MMDatasetBase
from machine_learning.dataset.parsers import VedaiParser, FlirAlignedParser
from machine_learning.utils.img import plot_imgs

data_cfg = load_cfg("/home/yangxf/WorkSpace/machine_learning/src/machine_learning/cfg/datasets/flir_aligned.yaml")
# parser = VedaiParser(data_cfg)
parser = FlirAlignedParser(data_cfg)
res = parser.parse()

dataset = MMDatasetBase(data=res["train"]["data"], labels=res["train"]["labels"], cache="ram", fraction=1)
data, label = dataset.get_data_and_label(5)
plot_imgs([data["img"], data["ir"]])
print(label)
