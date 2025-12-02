import time
from machine_learning.utils import load_cfg_from_yaml
from machine_learning.dataset import DatasetBase
from machine_learning.dataset.parsers import CocoTestParser
from machine_learning.utils.plots import visualize_img_with_bboxes
from machine_learning.utils.detection import yolo2voc

data_cfg = load_cfg_from_yaml(
    "/home/yangxf/WorkSpace/machine_learning/src/machine_learning/cfg/datasets/coco_test.yaml"
)
parser = CocoTestParser(data_cfg)
res = parser.parse()

dataset = DatasetBase(data=res["train"]["imgs"], labels=res["train"]["labels"], cache=None)

t_start = time.time()
image, label = dataset.get_data_and_label(10)
duration = time.time() - t_start
print(duration)

bboxes = label["label"][:, 1:5]
category_ids = label["label"][:, 0]
category_id_to_name = {x: str(x) for x in category_ids}
visualize_img_with_bboxes(
    image, yolo2voc(bboxes, image.shape[1], image.shape[0]), category_ids, category_id_to_name
)
