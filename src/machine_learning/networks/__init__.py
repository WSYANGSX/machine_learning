from .base import BaseNet
from .diffusion.unet import UNet
from .yolo import DarkNet53, V13Net
from .auto_encoder import AENet
from .gan import Generator, Discriminator
from .mmic import MMICNet

__all__ = ["BaseNet", "UNet", "DarkNet53", "AENet", "Generator", "Discriminator", "MMICNet"]

net_maps = {
    "auto_encoder": AENet,
    "gan": {"g": Generator, "d": Discriminator},
    "yolo_v3": DarkNet53,
    "yolo_v13": V13Net,
    "mmic": MMICNet,
}
