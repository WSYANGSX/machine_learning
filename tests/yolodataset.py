from machine_learning.utils import load_cfg
from machine_learning.dataset import YoloDataset
from machine_learning.dataset.parsers import CocoParser
from machine_learning.utils.img import plot_imgs

data_cfg = load_cfg("/home/yangxf/WorkSpace/machine_learning/src/machine_learning/cfg/datasets/coco-2017.yaml")
parser = CocoParser(data_cfg)
parsing = parser.parse()

hyp = {"mosaic": 0.8, "mixup": 0.1, "mask_ratio": 0.2, "mask_overlap": 0.3, "bgr": 0.5}

dataset = YoloDataset(imgs=parsing["train"]["data"], labels=parsing["train"]["labels"], cache=None, fraction=1, hyp=hyp)
label1 = dataset.get_data_and_label(1)
label2 = dataset.get_data_and_label(2)
label3 = dataset.get_data_and_label(3)
label4 = dataset.get_data_and_label(4)
label5 = dataset.get_data_and_label(5)
imgs = [label1["img"], label2["img"], label3["img"], label4["img"], label5["img"]]
plot_imgs(imgs, backend="pillow")
