from typing import Any

import math
import torch
import torch.nn as nn


class ModelEMA:
    """
    Updated Exponential Moving Average (EMA) implementation.

    Keeps a moving average of everything in the model state_dict (parameters and buffers).
    For EMA details see References.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, tau: int = 2000):
        self.model = model
        self.decay = lambda x: decay * (1 - math.exp(-x / tau)) if tau > 0 else decay
        self.shadow = {}
        self.shadow_buffers = {}
        self.backup = {}
        self.backup_buffers = {}
        self.updates = 0

        self.init_shadow()

    def init_shadow(self):
        # Initialize shadow weights
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

        # Initialize shadow buffers (e.g. BatchNorm running stats)
        for name, buffer in self.model.named_buffers():
            if buffer.dtype.is_floating_point:
                self.shadow_buffers[name] = buffer.data.clone().detach()

    def update(self, model=None):
        """Update EMA weights."""
        model = model or self.model
        self.updates += 1
        d = self.decay(self.updates)

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    self.shadow[name].mul_(d).add_(param.data, alpha=1 - d)

            for name, buffer in model.named_buffers():
                if buffer.dtype.is_floating_point and name in self.shadow_buffers:
                    self.shadow_buffers[name].mul_(d).add_(buffer.data, alpha=1 - d)

    def apply_shadow(self):
        """Apply EMA weights to the model."""
        self.backup = {}
        self.backup_buffers = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

        for name, buffer in self.model.named_buffers():
            if buffer.dtype.is_floating_point and name in self.shadow_buffers:
                self.backup_buffers[name] = buffer.data.clone()
                buffer.data.copy_(self.shadow_buffers[name])

    def restore(self):
        """Restore the original weights."""
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

        for name, buffer in self.model.named_buffers():
            if name in self.backup_buffers:
                buffer.data.copy_(self.backup_buffers[name])
        self.backup_buffers = {}

    def state_dict(self):
        """Get EMA states."""
        return {
            "shadow": {k: v.clone() for k, v in self.shadow.items()},
            "shadow_buffers": {k: v.clone() for k, v in self.shadow_buffers.items()},
            "updates": self.updates,
        }

    def load_state_dict(self, state_dict: dict[str, Any]):
        """Load EMA states."""
        device = self.model.device
        self.shadow = {k: v.to(device) for k, v in state_dict["shadow"].items()}
        self.shadow_buffers = {k: v.to(device) for k, v in state_dict.get("shadow_buffers", {}).items()}
        self.updates = state_dict.get("updates", 0)
