from machine_learning.utils import load_cfg
from machine_learning.data.dataset.datasets import MultiModalMasksDataset
from machine_learning.data.dataset.parsers import FireHoleParser
from machine_learning.utils.plots import plot_imgs
from machine_learning.utils.ops import img_tensor2np
from machine_learning.utils.segmentation import visualize_mask

data_cfg = load_cfg("/home/yangxf/WorkSpace/machine_learning/src/machine_learning/cfg/datasets/fire_hole.yaml")
parser = FireHoleParser(data_cfg)
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

dataset = MultiModalMasksDataset(
    data=res["train"]["data"],
    masks=res["train"]["labels"],
    nc=data_cfg["nc"],
    cache=False,
    fraction=1,
    rect=False,
    hyp=hyp,
    task="instance",
)

dataset.remove_item(100)
dataset.remove_item(500)
print(dataset.length)

for i in range(50):
    sample = dataset.__getitem__(i)
    img = sample["img"]
    ir = sample["ir"]
    masks = sample["masks"]
    cls = sample["cls"].numpy().reshape(-1)
    plot_imgs([img_tensor2np(img)])
    plot_imgs([img_tensor2np(ir)])

    visualize_mask(img_tensor2np(masks))

    print(sample["img_file"], sample["resized_shape"], sample["ori_shape"], cls)
    print(masks.shape)
