import numpy as np
from machine_learning.utils import load_cfg
from machine_learning.dataset import YoloDataset
from machine_learning.dataset.parsers import CocoParser
from machine_learning.utils.img import plot_imgs

np.set_printoptions(threshold=np.inf)

data_cfg = load_cfg("/home/yangxf/WorkSpace/machine_learning/src/machine_learning/cfg/datasets/coco-2017.yaml")
parser = CocoParser(data_cfg)
res = parser.parse()

cfg = load_cfg("/home/yangxf/WorkSpace/machine_learning/src/machine_learning/cfg/algorithms/yolo_v13.yaml")
data_cfg.update(cfg["data"])
print(data_cfg)

dataset = YoloDataset(
    imgs=res["train"]["imgs"],
    labels=res["train"]["labels"],
    cache=None,
    fraction=1,
    classes=[0, 32],
    hyp=data_cfg,
    single_cls=True,
    rect=True,
)
# sample1 = dataset.get_data_and_label(1)
# sample2 = dataset.get_data_and_label(2)
# sample3 = dataset.get_data_and_label(3)
# sample4 = dataset.get_data_and_label(4)
# sample5 = dataset.get_data_and_label(5)
# print(sample1["im_file"], sample1["cls"])
# print(sample2["im_file"], sample2["cls"])
# print(sample3["im_file"], sample3["cls"])
# print(sample4["im_file"], sample4["cls"])
# print(sample5["im_file"], sample5["cls"])
print(dataset.batch_shapes)
