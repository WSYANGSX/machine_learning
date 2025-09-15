from .base import DatasetBase
from .datasets import YoloDataset, MultimodalDataset
from .parsers import ParserBase, MinistParser, CocoParser, FlirParser, VedaiParser

__all__ = [
    "DatasetBase",
    "YoloDataset",
    "MultimodalDataset",
    "ParserBase",
    "MinistParser",
    "CocoParser",
    "FlirParser",
    "VedaiParser",
]

# parsers
PARSER_MAPS = {
    "minist": MinistParser,
    "coco-2017": CocoParser,
    "flir": FlirParser,
    "vedai": VedaiParser,
}

# datasets
DATASET_MAPS = {
    "basic": DatasetBase,
    "yolo": YoloDataset,
    "multi_modal": MultimodalDataset,
}
