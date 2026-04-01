from typing import Mapping, Any, Literal

from machine_learning.networks.base import BaseNet
from machine_learning.types.aliases import FilePath
from machine_learning.algorithms.segmentation import PerPixelSegmentationBase


class Unet(PerPixelSegmentationBase):
    def __init__(
        self,
        cfg: FilePath | Mapping[str, Any],
        net: BaseNet | None = None,
        name: str | None = None,
        device: Literal["cuda", "cpu", "auto"] = "auto",
        amp: bool = True,
        ema: bool = True,
        modality: str | None = "img",
    ) -> None:
        """
        Implementation of YoloV8 object detection algorithm

        Args:
            cfg (FilePath, Mapping[str, Any]): Configuration of the algorithm, it can be yaml file path or cfg dict.
            data (Mapping[str, Union[Dataset, Any]]): Parsed specific dataset data, must including train dataset and val
            dataset, may contain data information of the specific dataset.
            net (BaseNet): Models required by the YoloV8 algorithm.
            name (str): Name of the algorithm, it can be instantiated as v8, v9, v10, v11, v13 by cfg. Defaults to None.
            device (Literal["cuda", "cpu", "auto"], optional): Running device. Defaults to
            "auto"-automatic selection by algorithm.
            amp (bool): Whether to enable Automatic Mixed Precision. Defaults to False.
            ema (bool): Whether to enable Exponential Moving Average. Defaults to True.
            modality (str | None): The data modality to use for multimodal dataset selection. Only relevant for
            multimodal datasets. Defaults to "img".
        """
        super().__init__(cfg=cfg, net=net, name=name, device=device, amp=amp, ema=ema, modality=modality)

        # main parameters of the algorithm
        self.task = self.cfg["algorithm"]["task"]
        self.imgsz = self.cfg["algorithm"]["imgsz"]
        self.reg_max = self.cfg["algorithm"]["reg_max"]
        self.use_dfl = self.reg_max > 1
        self.close_mosaic_epoch = self.cfg["algorithm"]["close_mosaic_epoch"]
        self.max_det = self.cfg["algorithm"]["max_det"]
        self.single_cls = self.cfg["data"]["single_cls"]
        self.plot = self.cfg["algorithm"].get("plot", False)

        # threshold
        self.iou_thres = self.cfg["algorithm"]["iou_thres"]
        self.conf_thres_val = self.cfg["algorithm"]["conf_thres_val"]
        self.conf_thres_det = self.cfg["algorithm"]["conf_thres_det"]

        # weight
        self.box_weight = self.cfg["algorithm"].get("box")
        self.cls_weight = self.cfg["algorithm"].get("cls")
        self.dfl_weight = self.cfg["algorithm"].get("dfl")
