from dataclasses import dataclass, field


@dataclass
class TrainCfg:
    log_dir: str
    model_dir: str
    batch_size: int = field(default=256)
    data_num_workers: int = field(default=4)
    data_shuffle: bool = field(default=True)
    epochs: int = field(default=100)
    log_interval: int = field(default=10)
    save_interval: int = field(default=10)
