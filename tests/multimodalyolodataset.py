import cv2
import numpy as np
from machine_learning.utils import load_cfg
from machine_learning.data.dataset import YoloMultiModalDataset
from machine_learning.data.dataset.parsers import VedaiParser, FlirAlignedParser, MinistParser
from machine_learning.utils.plots import plot_imgs
from machine_learning.utils.detection import visualize_img_bboxes, yolo2voc

data_cfg = load_cfg("/home/yangxf/WorkSpace/machine_learning/src/machine_learning/cfg/datasets/flir_aligned.yaml")
parser = FlirAlignedParser(data_cfg)

# data_cfg = load_cfg("/home/yangxf/WorkSpace/machine_learning/src/machine_learning/cfg/datasets/minist.yaml")
# parser = MinistParser(data_cfg)

# data_cfg = load_cfg("/home/yangxf/WorkSpace/machine_learning/src/machine_learning/cfg/datasets/vedai_1024.yaml")
# parser = VedaiParser(data_cfg)

res = parser.parse()
# imgs = res["train"]["data"]
# irs = res["train"]["data"]
# res["train"]["data"] = {"imgs": imgs, "irs": irs}

dataset = YoloMultiModalDataset(
    imgs=res["train"]["data"],
    labels=res["train"]["labels"],
    nc=data_cfg["nc"],
    cache=True,
    fraction=1,
    modals=data_cfg["modals"],
    rect=True,
)

# print(dataset.modals)
# dataset.remove_item(100)
# dataset.remove_item(500)
# sample = dataset.get_sample(1)
# img = sample["img"]
# ir = sample["ir"]
# print(img.shape, ir.shape)
# cls = sample["cls"]
# plot_imgs([img, ir])

sample1 = dataset.get_sample(1)
sample2 = dataset.get_sample(2)
sample3 = dataset.get_sample(3)
sample4 = dataset.get_sample(4)
sample5 = dataset.get_sample(5)
print(sample1["img"].shape, sample1["ir"].shape)
print(sample2["img"].shape, sample2["ir"].shape)
print(sample3["img"].shape, sample3["ir"].shape)
print(sample4["img"].shape, sample4["ir"].shape)
print(sample5["img"].shape, sample5["ir"].shape)
