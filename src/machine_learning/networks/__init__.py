from .base import BaseNet
from .auto_encoder import AENet
from .diffusion.unet import UNet
from .gan import Generator, Discriminator
from .yolo import DarkNet53, V13Net, MMICNet, SuperYoloNet, COMONet, HyperMambaNet

__all__ = ["BaseNet", "UNet", "DarkNet53", "AENet", "Generator", "Discriminator", "MMICNet"]

NET_MAPS = {
    "auto_encoder": AENet,
    "gan": {"g": Generator, "d": Discriminator},
    "yolo_v3": DarkNet53,
    "yolo_v13": V13Net,
    "mmic": MMICNet,
    "super_yolo": SuperYoloNet,
    "como": COMONet,
    "hyper_mamba": HyperMambaNet,
}
