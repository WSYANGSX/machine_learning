from machine_learning.utils import load_cfg
from machine_learning.data.dataset import YoloDataset
from machine_learning.data.dataset.parsers import CocoParser
from machine_learning.utils.image import img_tensor2np
from machine_learning.utils.detection import visualize_img_bboxes, yolo2voc

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
    "bgr": 0.0,
}

dataset = YoloDataset(
    imgs=parsing["train"]["imgs"], labels=parsing["train"]["labels"], cache=None, fraction=0.5, hyp=hyp
)


for i in range(50):
    sample = dataset.__getitem__(i)
    img = sample["img"]
    bboxes = yolo2voc(sample["bboxes"], img.shape[2], img.shape[1])
    cls = sample["cls"].numpy().reshape(-1)
    visualize_img_bboxes(img_tensor2np(img), bboxes, cls, thickness=1)

    print(sample["im_file"], sample["resized_shape"], sample["ori_shape"])
