import torch
from PIL import Image
from machine_learning.utils.data_utils import LazyDataset, YoloParser, ParserCfg

parser_cfg = ParserCfg(
    dataset_dir="/home/yangxf/WorkSpace/machine_learning/data/coco-2017", labels=True, data_load_method="lazy"
)
yolo_parser = YoloParser(parser_cfg)
classes, train_img_paths, val_img_paths, train_labels_paths, val_labels_paths = yolo_parser.parse()

lazy_dataset = LazyDataset(train_img_paths, train_labels_paths, img_size=416)

index = torch.tensor(1)
img, bbs = lazy_dataset.__getitem__(index)
figure = Image.fromarray(img)
figure.show()
print(bbs)
