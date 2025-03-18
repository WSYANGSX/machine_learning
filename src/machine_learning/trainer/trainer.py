import os
import torch
import numpy as np
from typing import Sequence, Any
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
            data (Sequence[torch.Tensor | np.ndarray]): 数据集 (train_data, train_labels, val_data, val_labels)
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
        log_path = self.cfg.get(
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
            batch_size=self.cfg["batch_size"],
            shuffle=True,
            num_workers=self.cfg["data_num_workers"],
        )
        val_loader = DataLoader(
            validate_dataset,
            batch_size=self.cfg["batch_size"],
            shuffle=False,
            num_workers=self.cfg["data_num_workers"],
        )
        return train_loader, val_loader

    def train(self, start_epoch: int = 0) -> None:
        """完整训练"""
        print("[INFO] Start training...")

        for epoch in trange(start_epoch, self.cfg.get("epochs", 100)):
            train_loss = self._algorithm.train_epoch(epoch, self.writer, self.cfg.get("log_interval", 10))
            val_loss = self._algorithm.validate()

            # 学习率调整
            if self._algorithm._schedulers:
                for key, val in self._algorithm._schedulers.items():
                    if isinstance(val, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        val.step(val_loss[key])
                    else:
                        val.step()

            # 记录训练损失
            for key, val in train_loss.items():
                self.writer.add_scalar(f"{key} loss/train", val, epoch)

            # 记录验证损失
            for key, val in val_loss.items():
                self.writer.add_scalar(f"{key} loss/val", val, epoch)

            # 保存最佳模型
            if "save_metric" in val_loss:
                if val_loss["save_metric"] < self.best_loss:
                    self.best_loss = val_loss["save_metric"]
                    self.save_checkpoint(epoch, val_loss, self.best_loss, is_best=True)
            else:
                raise ValueError("val loss has no save metric.")

            # 定期保存
            if (epoch + 1) % self.cfg.get("save_interval", 10) == 0:
                self.save_checkpoint(epoch, val_loss, self.best_loss, is_best=False)

            # 打印日志
            print(f"Epoch: {epoch + 1:03d} | ", end="")
            for key, val in train_loss.items():
                print(f"{key} train loss {val:.4f} | ", end="")
            for key, val in val_loss.items():
                print(f"{key} val loss {val:.4f} | ", end="")
            for key, opt in self._algorithm._optimizers.items():
                print(f"{key} lr: {opt.param_groups[0]['lr']:.2e} | ")

    def train_from_checkpoint(self, checkpoint: str) -> None:
        point = self.load(checkpoint)

        self.cfg = point["cfg"]

        epoch = point["epoch"]

        self.train(epoch)

    def eval(self):
        self._algorithm.eval()

    def save_checkpoint(self, epoch: int, loss: dict, best_loss: float, is_best: bool = False) -> None:
        filename = f"checkpoint_epoch_{epoch}.pth"
        if is_best:
            filename = "best_model.pth"
        save_path = os.path.join(self.cfg["model_dir"], filename)

        self._algorithm.save(epoch, loss, best_loss, save_path)

    def load(self, checkpoint: str) -> dict[str:Any]:
        return self._algorithm.load(checkpoint)
