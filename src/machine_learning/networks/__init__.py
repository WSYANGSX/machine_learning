from .base import BaseNet
from .auto_encoder import AENet
from .diffusion.unet import UNet
from .gan import Generator, Discriminator
from .yolo import DarkNet53, V8Net, V13Net, M2I2HANet_v13, COMONet

__all__ = [
    "BaseNet",
    "UNet",
    "DarkNet53",
    "AENet",
    "Generator",
    "Discriminator",
    "M2I2HANet_v13",
    "V8Net",
    "V13Net",
    "COMONet",
]

NET_MAPS = {
    "auto_encoder": AENet,
    "gan": {"g": Generator, "d": Discriminator},
    "yolo_v3": DarkNet53,
    "yolo_v8": V8Net,
    "yolo_v13": V13Net,
    "m2i2ha": M2I2HANet_v13,
    "como": COMONet,
}
