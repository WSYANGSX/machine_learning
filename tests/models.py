import time
from machine_learning.utils.data_utils import YoloParser, ParserCfg

start = time.time()
parser_cfg = ParserCfg(dataset_dir="./data/coco-2017", labels=True, data_load_method="full")
yolo_parser = YoloParser(parser_cfg)
classes, train_img_paths, val_img_paths, train_labels_paths, val_labels_paths = yolo_parser.parse()
end = time.time()

print(end - start)
