from typing import Literal
from dataclasses import dataclass, field
from machine_learning.cfg.base import BaseCfg


@dataclass
class TrainerCfg(BaseCfg):
    # Set global random seed
    seed: int = field(default=23)

    # Training epochs
    epochs: int = field(default=100)

    # Batch intervals to write data
    log_interval: int = field(default=10)

    # Epoch intervals to save model
    save_interval: int = field(default=10)

    # Whether to save the best model
    save_best: bool = field(default=True)

    # Whether to enable Automatic Mixed Precision during training
    amp: bool = field(default=False)

    # Whether to enable Exponential Moving Average during training
    ema: bool = field(default=True)

    # Training device
    device: Literal["cpu", "cuda", "auto"] = field(default="auto")

    # Continue training from resume directory or specified .pth checkpoint file
    continue_training: bool = field(default=False)

    # Resume directory
    resume: str = field(default=None)

    # Specified .pth checkpoint file
    ckpt: str = field(default=None)
