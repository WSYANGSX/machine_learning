from PIL import Image
from machine_learning.utils import load_cfg
from machine_learning.dataset import DatasetBase
from machine_learning.dataset.parsers import MinistParser, CocoTestParser
from machine_learning.utils.img import plot_imgs


# minist
# data_cfg = load_cfg("/home/yangxf/WorkSpace/machine_learning/src/machine_learning/cfg/datasets/minist.yaml")
# parser = MinistParser(data_cfg)
# res = parser.parse()

# dataset = DatasetBase(data=res["train"][0], labels=res["train"][1])


# Image.fromarray(dataset.data[10]).show()
# print(dataset.labels[10])

data_cfg = load_cfg("/home/yangxf/WorkSpace/machine_learning/src/machine_learning/cfg/datasets/coco-test.yaml")
parser = CocoTestParser(data_cfg)
res = parser.parse()

dataset = DatasetBase(data=res["train"][0], labels=res["train"][1], cache="ram", fraction=1)
print(len(dataset.data), len(dataset.labels))

img1, _ = dataset.get_data_and_label(224)
imgs = [img1]
plot_imgs(imgs, backend="pillow")
