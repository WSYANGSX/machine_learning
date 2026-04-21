from dataclasses import dataclass, field
from machine_learning.cfg.base import BaseCfg


@dataclass
class TrainerCfg(BaseCfg):
    seed: int = field(default=23)
    epochs: int = field(default=100)
    log_interval: int = field(default=10)
    save_interval: int = field(default=10)
    save_best: bool = field(default=True)
    amp: bool = field(default=False)
    ema: bool = field(default=False)
    device: str = field(default="auto")
    continue_training: bool = field(default=False)
    ckpt: str = field(default=None)
