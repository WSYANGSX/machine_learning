import cv2
import numpy as np
from machine_learning.utils import load_cfg
from machine_learning.dataset import YoloMultiModalDataset
from machine_learning.dataset.parsers import VedaiParser, FlirAlignedParser, MinistParser
from machine_learning.utils.img import plot_imgs
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
print(dataset.modals)
dataset.remove_item(100)
dataset.remove_item(500)
sample = dataset.get_sample(1)
img = sample["img"]
ir = sample["ir"]
cls = sample["cls"]
cls = np.array(cls, dtype=np.int32).reshape(
    -1,
)
bbox = yolo2voc(sample["instances"].bboxes, img.shape[1], img.shape[0])
print(sample["img_file"])
visualize_img_bboxes(img=img, bboxes=bbox)
