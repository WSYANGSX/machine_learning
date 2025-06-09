from itertools import chain
from typing import Literal, Mapping
from machine_learning.models import BaseNet
from machine_learning.algorithms.base import AlgorithmBase
from machine_learning.utils import plot_raw_recon_figures

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class VAE(AlgorithmBase):
    def __init__(
        self,
        cfg: str,
        models: Mapping[str, BaseNet],
        name: str = "vae",
        device: Literal["cuda", "cpu", "auto"] = "auto",
    ) -> None:
        """
        变分自编码器的实现

        parameters:
        - cfg (str): 配置文件路径(YAML格式).
        - models (Mapping[str, BaseNet]): vae算法所需模型.{"encoder":model1,"decoder":model2}.
        - name (str): 算法名称. Default to "vae".
        - device (str): 运行设备(auto自动选择).
        """
        super().__init__(cfg, models, name, device)

        # -------------------- 配置优化器 --------------------
        self._configure_optimizers()
        self._configure_schedulers()

    def _configure_optimizers(self) -> None:
        opt_config = self.cfg["optimizer"]

        self.params = chain(self.models["encoder"].parameters(), self.models["decoder"].parameters())

        if opt_config["type"] == "Adam":
            self._optimizers.update(
                {
                    "vae": torch.optim.Adam(
                        params=self.params,
                        lr=opt_config["learning_rate"],
                        betas=(opt_config["beta1"], opt_config["beta2"]),
                        eps=opt_config["eps"],
                        weight_decay=opt_config["weight_decay"],
                    )
                }
            )
        else:
            ValueError(f"暂时不支持优化器:{opt_config['type']}")

    def _configure_schedulers(self) -> None:
        sch_config = self.cfg["scheduler"]

        if sch_config.get("type") == "ReduceLROnPlateau":
            self._schedulers.update(
                {
                    "vae": torch.optim.lr_scheduler.ReduceLROnPlateau(
                        self._optimizers["vae"],
                        mode="min",
                        factor=sch_config.get("factor", 0.1),
                        patience=sch_config.get("patience", 10),
                    )
                }
            )

    def train_epoch(self, epoch: int, writer: SummaryWriter, log_interval: int = 10) -> float:
        """训练单个epoch"""
        self.set_train()

        total_loss = 0.0
        criterion = nn.MSELoss()

        for batch_idx, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device, non_blocking=True)

            self._optimizers["vae"].zero_grad()

            mu, log_var = self._models["encoder"](data)
            std = torch.exp(0.5 * log_var)
            z = mu + std * torch.randn_like(mu)
            output = self._models["decoder"](z)

            kl_d = 0.5 * torch.sum(mu.pow(2) + log_var.exp() - log_var - 1, dim=1)
            loss = criterion(output, data) + kl_d.mean() * self.cfg["algorithm"]["beta"]
            loss.backward()  # 反向传播计算各权重的梯度

            torch.nn.utils.clip_grad_norm_(self.params, self.cfg["optimizer"]["grad_clip"])
            self._optimizers["vae"].step()

            total_loss += loss.item()

            if batch_idx % log_interval == 0:
                writer.add_scalar(
                    "loss/train_batch", loss.item(), epoch * len(self.train_loader) + batch_idx
                )  # batch loss
                writer.add_scalar(
                    "kl/train_batch", kl_d.mean().item(), epoch * len(self.train_loader) + batch_idx
                )  # batch kl

        avg_loss = total_loss / len(self.train_loader)

        return {"vae": avg_loss}

    def validate(self) -> float:
        """验证步骤"""
        self.set_eval()

        total_loss = 0.0
        criterion = nn.MSELoss()

        with torch.no_grad():
            for data, _ in self.val_loader:
                data = data.to(self.device, non_blocking=True)

                mu, log_var = self._models["encoder"](data)
                std = torch.exp(0.5 * log_var)
                z = mu + std * torch.randn_like(mu)
                output = self._models["decoder"](z)

                kl_d = 0.5 * torch.sum(
                    mu.pow(2) + log_var.exp() - log_var - 1, dim=1
                )  # 在处理损失时按照相同的损失处理方法，要么按照样本求和，要么按照样本平均，以保持两项在同一个量级上

                total_loss += (criterion(output, data) + kl_d.mean() * self.cfg["algorithm"]["beta"]).item()

        avg_loss = total_loss / len(self.val_loader)

        return {"vae": avg_loss, "save": avg_loss}

    def eval(self, num_samples: int = 5) -> None:
        """可视化重构结果"""
        self.set_eval()

        data, _ = next(iter(self.val_loader))
        sample_indices = torch.randint(low=0, high=len(data), size=(num_samples,))
        data = data[sample_indices].to(self.device)

        with torch.no_grad():
            mu, log_var = self._models["encoder"](data)
            # std = torch.exp(0.5 * log_var)
            # z = mu + std * torch.randn_like(mu)
            recons = self._models["decoder"](mu)

        plot_raw_recon_figures(data, recons)
