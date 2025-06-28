from dataclasses import dataclass, field


@dataclass
class TrainCfg:
    log_dir: str
    model_dir: str
    seed: int = field(default=23)
    epochs: int = field(default=100)
    log_interval: int = field(default=10)
    save_interval: int = field(default=10)
    save_best: bool = field(default=True)
