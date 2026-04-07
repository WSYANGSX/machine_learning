import cv2
import torch
import numpy as np
from machine_learning.utils import load_cfg
from machine_learning.data.dataset import MultiModalYoloDataset
from machine_learning.data.dataset.parsers import VedaiParser, FlirAlignedParser, DVParser
from machine_learning.utils.plots import plot_imgs
from machine_learning.utils.detection import visualize_img_bboxes, yolo2voc
from machine_learning.utils.ops import img_tensor2np

data_cfg = load_cfg("/home/yangxf/WorkSpace/machine_learning/src/machine_learning/cfg/datasets/flir_aligned.yaml")
parser = FlirAlignedParser(data_cfg)

# data_cfg = load_cfg("/home/yangxf/WorkSpace/machine_learning/src/machine_learning/cfg/datasets/drone_vehicle.yaml")
# parser = DVParser(data_cfg)

# data_cfg = load_cfg("/home/yangxf/WorkSpace/machine_learning/src/machine_learning/cfg/datasets/vedai_1024.yaml")
# parser = VedaiParser(data_cfg)

res = parser.parse()
# res["train"]["data"] = {"imgs": imgs, "irs": irs}

hyp = {
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "cutmix": 0.0,
    "erasing": 0.4,
    "crop_fraction": 1.0,
    "copy_paste_mode": "flip",
    "auto_augment": "randaugment",
    "flipud": 0.0,
    "fliplr": 0.5,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "perspective": 0.0,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "bgr": 0.0,
}

dataset = MultiModalYoloDataset(
    data=res["train"]["data"],
    labels=res["train"]["labels"],
    nc=data_cfg["nc"],
    cache=False,
    fraction=1,
    modalities=data_cfg["modalities"],
    rect=False,
    hyp=hyp,
)

dataset.remove_item(100)
dataset.remove_item(500)
print(dataset.length)

for i in range(50):
    sample = dataset.__getitem__(i)
    img = sample["img"]
    ir: torch.Tensor = sample["ir"]
    bboxes = yolo2voc(sample["bboxes"], img.shape[2], img.shape[1])
    cls = sample["cls"].numpy().reshape(-1)
    visualize_img_bboxes(img_tensor2np(img), np.zeros((0, 4), dtype=np.float32), [], thickness=1)
    plot_imgs([img_tensor2np(ir)])

    print(sample["img_file"], sample["resized_shape"], sample["ori_shape"])
    print(sample["ir_file"], sample["resized_shape"], sample["ori_shape"])
