from dataclasses import dataclass, field


@dataclass
class TrainCfg:
    log_dir: str
    model_dir: str
    data_num_workers: int = field(default=4)
    epochs: int = field(default=100)
    log_interval: int = field(default=10)
    save_interval: int = field(default=10)
    batch_size: int = field(default=256)
