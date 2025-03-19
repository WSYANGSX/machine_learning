from abc import ABC, abstractmethod

import torch.nn as nn


class BaseNet(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def _initialize_weights(self):
        pass

    @abstractmethod
    def view_structure(self):
        pass
