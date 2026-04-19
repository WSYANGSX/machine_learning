from machine_learning.utils import load_cfg
from machine_learning.dataset import YoloDataset
from machine_learning.dataset.parsers import CocoParser
from machine_learning.utils.plots import plot_imgs

data_cfg = load_cfg("/home/yangxf/WorkSpace/machine_learning/src/machine_learning/cfg/datasets/coco-2017.yaml")
parser = CocoParser(data_cfg)
parsing = parser.parse()

hyp = {
    "mosaic": 1.0,
    "mixup": 0.2,
    "copy_paste": 0.6,
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
}

dataset = YoloDataset(imgs=parsing["train"]["imgs"], labels=parsing["train"]["labels"], cache=None, fraction=1, hyp=hyp)
label1 = dataset.get_sample(1)
label2 = dataset.get_sample(2)
label3 = dataset.get_sample(3)
label4 = dataset.get_sample(4)
label5 = dataset.get_sample(5)
print(
    label1["img"].shape,
    label2["img"].shape,
    label3["img"].shape,
    label4["img"].shape,
    label5["img"].shape,
)
