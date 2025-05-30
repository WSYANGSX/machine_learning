import os
import torch
from typing import Any, Union
from tqdm import trange

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .trainer_cfg import TrainCfg
from machine_learning.algorithms import AlgorithmBase
from machine_learning.utils.data_utils import FullDataset, LazyDataset
from machine_learning.utils.others import set_seed


class Trainer:
    def __init__(
        self,
        cfg: TrainCfg,
        datasets: dict[str, Union[FullDataset, LazyDataset]],
        algo: AlgorithmBase,
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

        # ------------------ 配置随机种子 --------------------
        set_seed(self.cfg.seed)
        print(f"[INFO] Current seed: {self.cfg.seed}")

        # -------------------- 配置数据 --------------------
        self.batch_size = self.cfg.batch_size
        self._configure_dataloader(train_dataset=datasets["train"], val_dataset=datasets["val"])

        # -------------------- 配置记录器 --------------------
        self._configure_writer()
        self.best_loss = torch.inf

    def _configure_dataloader(
        self, train_dataset: Union[FullDataset, LazyDataset], val_dataset: Union[FullDataset, LazyDataset]
    ):
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=self.cfg.data_shuffle,
            num_workers=self.cfg.data_num_workers,
            collate_fn=train_dataset.collate_fn if hasattr(train_dataset, "collate_fn") else None,
        )
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.cfg.data_num_workers,
            collate_fn=val_dataset.collate_fn if hasattr(val_dataset, "collate_fn") else None,
        )
        self._algorithm._initialize_data_loader(
            train_loader=train_loader, val_loader=val_loader, batch_size=self.cfg.batch_size
        )

    def _configure_writer(self):
        log_path = self.cfg.log_dir

        log_path = os.path.abspath(log_path)

        try:
            os.makedirs(log_path, exist_ok=True)
        except OSError as e:
            raise RuntimeError(f"Failed to create log directory at {log_path}: {e}")

        self.writer = SummaryWriter(log_dir=log_path)

    def train(self, start_epoch: int = 0) -> None:
        """完整训练"""
        print("[INFO] Start training...")

        for epoch in trange(start_epoch, self.cfg.epochs):
            train_loss = self._algorithm.train_epoch(epoch, self.writer, self.cfg.log_interval)
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
            if (
                self.cfg.save_best and "save" in val_loss
            ):  # 必须在train_cfg中配置保存best_model选项，同时在val_loss中返回“save_loss”
                if val_loss["save"] < self.best_loss:
                    self.best_loss = val_loss["save"]
                    self.save_checkpoint(epoch, val_loss, self.best_loss, is_best=True)
            else:
                print("Saving of the best loss model skipped.")

            # 定期保存
            if (epoch + 1) % self.cfg.save_interval == 0:
                self.save_checkpoint(epoch, val_loss, self.best_loss, is_best=False)

            # 打印日志
            print(f"Epoch: {epoch + 1:03d} | ", end="")
            for key, val in train_loss.items():
                print(f"{key} train loss {val:.4f} | ", end="")
            for key, val in val_loss.items():
                if key != "save":
                    print(f"{key} val loss {val:.4f} | ", end="")
                else:
                    print(f"{key} loss {val:.4f} | ", end="")
            for key, opt in self._algorithm._optimizers.items():
                print(f"{key} lr: {opt.param_groups[0]['lr']:.2e} | ")

    def train_from_checkpoint(self, checkpoint: str) -> None:
        state_dict = self.load(checkpoint)
        self.cfg = state_dict["cfg"]
        epoch = state_dict["epoch"]
        self.train(epoch)

    def eval(self, num_samples: int = 5):
        self._algorithm.eval(num_samples)

    def save_checkpoint(self, epoch: int, loss: dict, best_loss: float, is_best: bool = False) -> None:
        model_path = self.cfg.model_dir

        model_path = os.path.abspath(model_path)

        try:
            os.makedirs(model_path, exist_ok=True)
        except OSError as e:
            raise RuntimeError(f"Failed to create log directory at {model_path}: {e}")

        filename = f"checkpoint_epoch_{epoch}.pth"
        if is_best:
            filename = "best_model.pth"
        save_path = os.path.join(model_path, filename)

        self._algorithm.save(epoch, loss, best_loss, save_path)

    def load(self, checkpoint: str) -> dict[str:Any]:
        return self._algorithm.load(checkpoint)
