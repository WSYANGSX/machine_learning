# 对于无监督式学习，比较好的办法是重建自己，通过重建数据发现数据的模态特征信息
# auto-encoder相当于对数据进行降维处理，类似PCA，只不过PCA是通过求解特征向量进行降维，是线性降维方式，而auto-encoder是非线性降维方式
import os
from tqdm import trange
from itertools import chain
from typing import Literal, Mapping

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from machine_learning.models import BaseNet
from machine_learning.algorithms.base import AlgorithmBase


class AutoEncoder(AlgorithmBase):
    def __init__(
        self,
        cfg: str,
        models: Mapping[str, BaseNet],
        name: str = "auto_encoder",
        device: Literal["cuda", "cpu", "auto"] = "auto",
    ) -> None:
        """
        自编码器实现

        parameters:
        - config_file (str): 配置文件路径(YAML格式).
        - models (Mapping[str, BaseNet]): auto-encoder算法所需模型.{"encoder":model1,"decoder":model2}.
        - name (str): 算法名称. Default to "auto_encoder".
        - device (str): 运行设备(auto自动选择).
        """
        super().__init__(cfg=cfg, models=models, name=name, device=device)

        # -------------------- 配置优化器 --------------------
        self._configure_optimizer()
        self._configure_scheduler()

    def _configure_optimizer(self) -> None:
        opt_config = self.config["optimizer"]

        params = chain(self._models["encoder"].parameters(), self._models["decoder"].parameters())

        if opt_config["type"] == "Adam":
            self.optimizer = torch.optim.Adam(
                params=params,
                lr=opt_config["learning_rate"],
                betas=(opt_config["beta1"], opt_config["beta2"]),
                eps=opt_config["eps"],
                weight_decay=opt_config["weight_decay"],
            )
        elif opt_config["type"] == "SGD":
            self.optimizer = torch.optim.SGD(
                params=params,
                lr=opt_config["learning_rate"],
                momentum=opt_config["momentum"],
                dampening=opt_config["dampening"],
                weight_decay=opt_config["weight_decay"],
            )
        else:
            ValueError(f"暂时不支持优化器:{opt_config['type']}")

    def _configure_scheduler(self) -> None:
        self.scheduler = None
        sched_config = self.config.get("scheduler", {})
        if sched_config.get("type") == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=sched_config.get("factor", 0.1),
                patience=sched_config.get("patience", 10),
            )

    def train_epoch(self, epoch: int, writer: SummaryWriter) -> float:
        """训练单个epoch"""
        self._models["encoder"].train()
        self._models["decoder"].train()

        total_loss = 0.0
        criterion = nn.MSELoss()

        for batch_idx, (data, _) in enumerate(self.train_loader):
            data = data.to(self._device, non_blocking=True)

            self.optimizer.zero_grad()

            z = self.encoder(data)
            output = self.decoder(z)

            loss = criterion(output, data)
            loss.backward()  # 反向传播计算各权重的梯度

            torch.nn.utils.clip_grad_norm_(self.parameters(), self.config["training"]["grad_clip"])
            self.optimizer.step()

            total_loss += loss.item()

            if batch_idx % self.config["logging"].get("log_interval", 10) == 0:
                writer.add_scalar("Loss/train_batch", loss.item(), epoch * len(self.train_loader))  # batch loss

        avg_loss = total_loss / len(self.train_loader)
        writer.add_scalar("Loss/train", avg_loss, epoch)  # epoch loss
        return avg_loss

    def validate(self) -> float:
        """验证步骤"""
        self.models["encoder"].eval()
        self.models["decoder"].eval()

        total_loss = 0.0
        criterion = nn.MSELoss()

        with torch.no_grad():
            for data, _ in self.val_loader:
                data = data.to(self._device, non_blocking=True)
                recon = self.decoder(self.encoder(data))
                total_loss += criterion(recon, data).item()

        avg_loss = total_loss / len(self.val_loader)
        return avg_loss

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """保存模型检查点"""
        state = {
            "epoch": epoch,
            "encoder_state": self.encoder.state_dict(),
            "decoder_state": self.decoder.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
            "config": self.config,
        }

        filename = f"checkpoint_epoch_{epoch}.pth"
        if is_best:
            filename = "best_model.pth"

        save_path = os.path.join(self.config["logging"]["model_dir"], filename)
        torch.save(state, save_path)
        print(f"Saved checkpoint to {save_path}")

    def visualize_reconstruction(self, num_samples: int = 5) -> None:
        """可视化重构结果"""
        self.decoder.eval()
        self.encoder.eval()

        data, _ = next(iter(self.validate_loader))
        sample_indices = torch.randint(low=0, high=len(data), size=(num_samples,))
        data = data[sample_indices].to(self.device)

        with torch.no_grad():
            reconstructions = self.decoder(self.encoder(data))

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 4))
        for i in range(num_samples):
            # 原始图像
            ax = plt.subplot(2, num_samples, i + 1)
            plt.imshow(data[i].cpu().squeeze(), cmap="gray")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # 重构图像
            ax = plt.subplot(2, num_samples, i + 1 + num_samples)
            plt.imshow(reconstructions[i].cpu().squeeze(), cmap="gray")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()
