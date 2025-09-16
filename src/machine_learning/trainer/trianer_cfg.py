from dataclasses import dataclass, field, MISSING

from machine_learning.cfg.base import BaseCfg


@dataclass
class TrainerCfg(BaseCfg):
    log_dir: str = MISSING
    ckpt_dir: str = MISSING
    seed: int = field(default=23)
    epochs: int = field(default=100)
    log_interval: int = field(default=10)
    save_interval: int = field(default=10)
    save_best: bool = field(default=True)
