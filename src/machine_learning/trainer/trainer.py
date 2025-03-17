import os
import torch
import torch.nn as nn  # noqa:F401
import numpy as np
from typing import Sequence
from tqdm import trange

from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from machine_learning.utils import CustomDataset
from machine_learning.algorithms import AlgorithmBase


class Trainer:
    def __init__(
        self, cfg: dict, data: Sequence[torch.Tensor | np.ndarray], transform: transforms.Compose, algo: AlgorithmBase
    ):
        """机器学习算法训练器.

        Args:
            cfg (dict): 训练器配置信息.
            data (Sequence[torch.Tensor  |  np.ndarray]): 数据集 (train_data, train_labels, val_data, val_labels)
            transform (transforms.Compose): 数据转换器.
            algo (AlgorithmBase): 算法.
        """
        self.cfg = cfg
        self._algorithm = algo

        # -------------------- 配置数据 --------------------
        train_loader, val_loader = self._load_datasets(*data, transform)
        self._algorithm._initialize_data_loader(train_loader, val_loader)

        # -------------------- 配置记录器 --------------------
        self._configure_writer()
        self.best_loss = float("inf")

    def _configure_writer(self):
        log_path = self.cfg["logging"].get(
            "log_dir",
            os.path.join(
                os.getcwd(),
                "logs",
                self._algorithm.name,
            ),
        )

        log_path = os.path.abspath(log_path)

        try:
            os.makedirs(log_path, exist_ok=True)
        except OSError as e:
            raise RuntimeError(f"Failed to create log directory at {log_path}: {e}")

        self.writer = SummaryWriter(log_dir=log_path)

    def _load_datasets(
        self,
        train_data: torch.Tensor | np.ndarray,
        train_labels: torch.Tensor | np.ndarray,
        val_data: torch.Tensor | np.ndarray,
        val_labels: torch.Tensor | np.ndarray,
        transform: Compose,
    ):
        # 创建dataset和datasetloader
        train_dataset = CustomDataset(train_data, train_labels, transform)
        validate_dataset = CustomDataset(val_data, val_labels, transform)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg["training"]["batch_size"],
            shuffle=True,
            num_workers=self.cfg["data"]["num_workers"],
        )
        val_loader = DataLoader(
            validate_dataset,
            batch_size=self.cfg["training"]["batch_size"],
            shuffle=False,
            num_workers=self.cfg["data"]["num_workers"],
        )
        return train_loader, val_loader

    def train_model(self) -> None:
        """完整训练"""
        print("[INFO] Start training...")
        for epoch in trange(self.cfg["training"]["epochs"]):
            train_loss, info = self._algorithm.train_epoch(epoch)
            val_loss = self._algorithm.validate()

            # 学习率调整
            if self._algorithm.scheduler:
                if isinstance(self._algorithm.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self._algorithm.scheduler.step(val_loss)
                else:
                    self._algorithm.scheduler.step()

            # 记录验证损失
            self.writer.add_scalar("Loss/val", val_loss, epoch)

            # 保存最佳模型
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)

            # 定期保存
            if (epoch + 1) % self.cfg["training"]["save_interval"] == 0:
                self.save_checkpoint(epoch)

            # 打印日志
            print(info)
