from typing import Any, Mapping

import os

from .base import AlgorithmBase
from .generation import AutoEncoder, VAE, GAN, VQ_VAE, Diffusion
from .detection import YoloV3, YoloV8, MultimodalDetection
from .segmentation import (
    YoloV8Segmentation,
    MaskFormer,
    Mask2Former,
    PerPixelSegmentation,
    MaskSegmentation,
    MultimodalPerPixelSegmentation,
    MultimodalMaskSegmentation,
)
from machine_learning.utils import load_cfg
from machine_learning.utils.constants import ALGOCFG_PATH


__all__ = [
    "AlgorithmBase",
    "AutoEncoder",
    "VAE",
    "GAN",
    "Diffusion",
    "VQ_VAE",
    "YoloV3",
    "YoloV8",
    "YoloV13",
    "MultimodalDetection",
    "YoloV8Segmentation",
    "MaskFormer",
    "Mask2Former",
    "PerPixelSegmentation",
    "MaskSegmentation",
    "MultimodalPerPixelSegmentation",
    "MultimodalMaskSegmentation",
]


def get_alogrithm_cfg(cfg: str | Mapping[str, Any], overwrites: dict[str, Any] | None = None) -> dict:
    """Load algorithm cfg from file or dict."""
    if isinstance(cfg, str):
        cfg = os.path.join(ALGOCFG_PATH, cfg)
    cfg = load_cfg(cfg)

    for k, v in cfg.items():
        if isinstance(v, dict):
            cfg[k] = get_alogrithm_cfg(v, overwrites)
        else:
            if overwrites is not None and k in overwrites:
                cfg[k] = overwrites[k]

    return cfg


# Algorithm factory function
class AlgorithmFactory:
    """A factory class for creating algorithms."""

    def __init__(self) -> None:
        self.algorithms = {}

    def register_algorithm(self, name: str, algorithm_class: AlgorithmBase) -> None:
        """Register an algorithm class."""
        self.algorithms[name] = algorithm_class

    def create_algorithm(self, algo: str, *args, **kwargs) -> AlgorithmBase:
        if algo not in self.algorithms:
            raise ValueError(f"Unknown algorithm: {algo}")

        return self.algorithms[algo](*args, **kwargs)


global_factory = AlgorithmFactory()

# Regist algorithms
# generation algorithms
global_factory.register_algorithm("vae", VAE)
global_factory.register_algorithm("gan", GAN)
global_factory.register_algorithm("vq_vae", VQ_VAE)
global_factory.register_algorithm("diffusion", Diffusion)
global_factory.register_algorithm("autoencoder", AutoEncoder)

# object detection algorithms
global_factory.register_algorithm("yolo_v3", YoloV3)
global_factory.register_algorithm("yolo_v8", YoloV8)
global_factory.register_algorithm("yolo_v13", YoloV8)
global_factory.register_algorithm("m2i2ha", MultimodalDetection)
global_factory.register_algorithm("como", MultimodalDetection)

# segmentation algorithms
global_factory.register_algorithm("unet", PerPixelSegmentation)
global_factory.register_algorithm("fghf", PerPixelSegmentation)
global_factory.register_algorithm("mask_segmentation", MaskSegmentation)
global_factory.register_algorithm("yolov8_segmentation", YoloV8Segmentation)
global_factory.register_algorithm("maskformer", MaskFormer)
global_factory.register_algorithm("mask2former", Mask2Former)
global_factory.register_algorithm("multimodal_per_pixel_segmentation", MultimodalPerPixelSegmentation)
global_factory.register_algorithm("multimodal_mask_segmentation", MultimodalMaskSegmentation)
