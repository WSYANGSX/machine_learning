from abc import ABC

import torch
import torch.nn as nn

from machine_learning.utils.logger import LOGGER


class BaseNet(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _initialize_weights(self):
        LOGGER.info(f"Initializing weights of {self.__class__.__name__} with Kaiming normal...")

        for module in self.modules():
            if isinstance(
                module,
                (
                    nn.Conv1d,
                    nn.Conv2d,
                    nn.Conv3d,
                    nn.ConvTranspose1d,
                    nn.ConvTranspose2d,
                    nn.ConvTranspose3d,
                    nn.Linear,
                ),
            ):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def view_structure(self):
        LOGGER.info(f"{self.__class__.__name__} structure:")
