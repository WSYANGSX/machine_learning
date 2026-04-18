from machine_learning.utils import load_cfg
from machine_learning.data.dataset.datasets import SemanticMaskDataset
from machine_learning.data.dataset.parsers import SBDParser
from machine_learning.utils.plots import plot_imgs
from machine_learning.utils.ops import img_tensor2np
from machine_learning.utils.segment import visualize_mask, generate_gt_edges
import torch

torch.set_printoptions(threshold=torch.inf)


data_cfg = load_cfg("/home/yangxf/WorkSpace/machine_learning/src/machine_learning/cfg/datasets/sbd.yaml")
parser = SBDParser(data_cfg)
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

dataset = SemanticMaskDataset(
    imgs=res["train"]["imgs"],
    masks=res["train"]["labels"],
    nc=data_cfg["nc"],
    cache=False,
    fraction=1,
    rect=False,
    hyp=hyp,
    augment=True,
)


for i in range(50):
    sample = dataset.__getitem__(i)
    img = sample["img"]
    masks = sample["mask"]
    edge2 = generate_gt_edges(masks, edge_width=3)

    cls = sample["cls"].numpy().reshape(-1)
    plot_imgs(
        [
            img_tensor2np(img),
            img_tensor2np(masks),
        ]
    )
    plot_imgs([img_tensor2np(edge2)], cmap="gray")
    print(cls)
    print(sample["img_file"], sample["resized_shape"], sample["ori_shape"])
