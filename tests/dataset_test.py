from pympler import asizeof
from PIL import Image
from machine_learning.utils import load_cfg_from_yaml
from machine_learning.dataset import DatasetBase
from machine_learning.dataset.parsers import MinistParser, CocoTestParser
from machine_learning.utils.img import plot_imgs

data_cfg = load_cfg_from_yaml(
    "/home/yangxf/WorkSpace/machine_learning/src/machine_learning/cfg/datasets/coco_test.yaml"
)
parser = CocoTestParser(data_cfg)
res = parser.parse()

dataset = DatasetBase(data=res["train"]["imgs"], labels=res["train"]["labels"], cache=None, fraction=1)
img1, _ = dataset.get_data_and_label(1)
img2, _ = dataset.get_data_and_label(2)
img3, _ = dataset.get_data_and_label(3)
img4, _ = dataset.get_data_and_label(4)
img5, _ = dataset.get_data_and_label(5)
imgs = [img1, img2, img3, img4, img5]
plot_imgs(imgs, backend="pillow")

# data_cfg = load_cfg_from_yaml("/home/yangxf/WorkSpace/machine_learning/src/machine_learning/cfg/datasets/minist.yaml")
# parser = MinistParser(data_cfg)
# res = parser.parse()

# dataset = DatasetBase(data=res["train"]["imgs"], labels=res["train"]["labels"])


# Image.fromarray(dataset.data[512]).show()
# print(dataset.labels[512])
