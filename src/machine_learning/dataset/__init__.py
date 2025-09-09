from .dataset import DatasetBase, ImgDataset, MultimodalDataset
from .parsers import ParserBase, MinistParser, YoloParser, FlirParser, VedaiParser

PARSER_MAPS = {
    "minist": MinistParser,
    "coco-2017": YoloParser,
    "flir": FlirParser,
    "vedai": VedaiParser,
}
