from typing import Literal, Any

import numpy as np
from torch.utils.data import Dataset, DataLoader

from .base import DatasetBase
from .datasets import YoloDataset, MultimodalDataset
from .parsers import ParserBase, MinistParser, CocoParser, FlirParser, VedaiParser

__all__ = [
    "DatasetBase",
    "YoloDataset",
    "MultimodalDataset",
    "ParserBase",
    "MinistParser",
    "CocoParser",
    "FlirParser",
    "VedaiParser",
]

# parsers
PARSER_MAPS = {
    "minist": MinistParser,
    "coco-2017": CocoParser,
    "flir": FlirParser,
    "vedai": VedaiParser,
}

# datasets
DATASET_MAPS = {
    "basic": DatasetBase,
    "yolo": YoloDataset,
    "multi_modal": MultimodalDataset,
}


def build_dataset(
    type: str,
    cfg: dict[str, Any],
    data: np.ndarray | list[str],
    labels: np.ndarray | list[str],
    batch_size: int,
    mode: Literal["train", "val", "test"],
) -> Dataset:
    """Build Dataset.

    Args:
        type (str): The dataset type.
        cfg (dict[str, Any]): The dataset configure.
        data (np.ndarray | list[str]): The data of the dataset.
        data (np.ndarray | list[str]): The labels of the dataset.
        batch_size (int): The batch size of the dataset.
        mode (Literal[&quot;train&quot;, &quot;val&quot;, &quot;test&quot;]): The mode of the dataset.

    Returns:
        Dataset: The builded dataset.
    """
    augment = cfg["augment"] if cfg.get("augment") is not None else mode == "train"
    cache = cfg.get("cache")
    fraction = cfg.get("fraction", 1.0) if mode == "trian" else 1.0

    if type == "DatasetBase":
        return DatasetBase(
            data=data,
            labels=labels,
            cache=cache,
            augment=augment,
            hyp=cfg,
            batch_size=batch_size,
            fraction=fraction,
            mode=mode,
        )
    elif type == "YoloDataset":
        return YoloDataset(
            imgs=data,
            labels=labels,
            imgsz=cfg.get("imgsz", 640),
            task=cfg["task"],
            rect=cfg["rect"],
            stride=cfg.get("stride", 32),
            pad=0.0 if mode == "train" else 0.5,
            single_cls=cfg.get("single_cls", False),
            classes=cfg.get("class", None),
            cache=cache,
            augment=augment,
            hyp=cfg,
            batch_size=batch_size,
            fraction=fraction,
            mode=mode,
        )
    elif type == "multimodal":
        ...


def build_dataloader(dataset: Dataset, batch: int, workers: int, shuffle: bool) -> DataLoader:
    """Return an InfiniteDataLoader or DataLoader for train or val set."""
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min(os.cpu_count() // max(nd, 1), workers)  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return InfiniteDataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=PIN_MEMORY,
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
    )
