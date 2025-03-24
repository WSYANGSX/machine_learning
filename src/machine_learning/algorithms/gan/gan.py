from typing import Literal, Mapping

import torch

from machine_learning.models import BaseNet
from machine_learning.algorithms.base import AlgorithmBase
from machine_learning.utils import plot_figures


class GAN(AlgorithmBase):
    def __init__(
        self,
        cfg: str,
        models: Mapping[str, BaseNet],
        name: str = "gan",
        device: Literal["cuda", "cpu", "auto"] = "auto",
    ) -> None:
        """
        生成对抗网络实现

        parameters:
        - cfg (str): 配置文件路径(YAML格式).
        - models (Mapping[str, BaseNet]): gan算法所需模型.{"generator": model1, "discriminator": model2}.
        - name (str): 算法名称. Default to "gan".
        - device (str): 运行设备(auto自动选择).
        """
        super().__init__(cfg, models, name, device)

        # -------------------- 配置优化器 -------------------
        self._configure_optimizers()
        self._configure_schedulers()

        # -------------------- 先验 ------------------------
        self.z_dim = self.models["generator"].input_dim

    def _configure_optimizers(self) -> None:
        opt_config = self.cfg["optimizer"]

        if opt_config["type"] == "Adam":
            self._optimizers.update(
                {
                    "generator": torch.optim.Adam(
                        params=self.models["generator"].parameters(),
                        lr=opt_config["g_learning_rate"],
                        betas=(opt_config["g_beta1"], opt_config["g_beta2"]),
                        eps=opt_config["g_eps"],
                        weight_decay=opt_config["g_weight_decay"],
                    )
                }
            )
            self._optimizers.update(
                {
                    "discriminator": torch.optim.Adam(
                        params=self.models["discriminator"].parameters(),
                        lr=opt_config["d_learning_rate"],
                        betas=(opt_config["d_beta1"], opt_config["d_beta2"]),
                        eps=opt_config["d_eps"],
                        weight_decay=opt_config["d_weight_decay"],
                    )
                }
            )
        else:
            ValueError(f"暂时不支持优化器:{opt_config['type']}")

    def _configure_schedulers(self) -> None:
        sched_config = self.cfg["scheduler"]

        if sched_config.get("type") == "StepLR":
            self._schedulers.update(
                {"generator": torch.optim.lr_scheduler.StepLR(self._optimizers["generator"], step_size=30, gamma=0.1)}
            )
            self._schedulers.update(
                {
                    "discriminator": torch.optim.lr_scheduler.StepLR(
                        self._optimizers["discriminator"], step_size=30, gamma=0.1
                    )
                }
            )

    def train_discriminator(self) -> float:
        """训练单个discriminator epoch"""
        self.models["generator"].eval()
        self.models["discriminator"].train()

        total_loss = 0.0

        for _, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device, non_blocking=True)

            z_prior = torch.randn((len(data), self.z_dim), device=self.device, dtype=torch.float32)
            data_ = self.models["generator"](z_prior)

            self._optimizers["discriminator"].zero_grad()

            real_preds = self.models["discriminator"](data)
            fake_preds = self.models["discriminator"](data_)

            loss = discriminator_criterion(real_preds, fake_preds)
            loss.backward()  # 反向传播计算各权重的梯度

            torch.nn.utils.clip_grad_norm_(
                self.models["discriminator"].parameters(), self.cfg["training"]["grad_clip"]["discriminator"]
            )
            self._optimizers["discriminator"].step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)

        return avg_loss

    def eval_discriminator(self) -> float:
        self.models["discriminator"].eval()
        self.models["generator"].eval()

        val_total_loss = 0.0

        with torch.no_grad():
            for data, _ in self.val_loader:
                data = data.to(self.device, non_blocking=True)

                z_prior = torch.randn((len(data), self.z_dim), device=self.device, dtype=torch.float32)
                data_ = self.models["generator"](z_prior)

                real_preds = self.models["discriminator"](data)
                fake_preds = self.models["discriminator"](data_)

                loss = discriminator_criterion(real_preds, fake_preds)
                val_total_loss += loss.item()

        avg_loss = val_total_loss / len(self.val_loader)

        return avg_loss

    def train_generator(self):
        """训练单个generator epoch"""
        self.models["generator"].train()
        self.models["discriminator"].eval()

        total_loss = 0.0

        for _, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device, non_blocking=True)

            z_prior = torch.randn((len(data), self.z_dim), device=self.device, dtype=torch.float32)
            data_ = self.models["generator"](z_prior)

            self._optimizers["generator"].zero_grad()

            fake_preds = self.models["discriminator"](data_)

            loss = generator_criterion(fake_preds)
            loss.backward()  # 反向传播计算各权重的梯度

            torch.nn.utils.clip_grad_norm_(
                self.models["generator"].parameters(), self.cfg["training"]["grad_clip"]["generator"]
            )
            self._optimizers["generator"].step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)

        return avg_loss

    def eval_generator(self) -> float:
        self.models["generator"].eval()
        self.models["discriminator"].eval()

        val_total_loss = 0.0

        with torch.no_grad():
            for data, _ in self.val_loader:
                data = data.to(self.device, non_blocking=True)

                z_prior = torch.randn((len(data), self.z_dim), device=self.device, dtype=torch.float32)
                data_ = self.models["generator"](z_prior)

                fake_preds = self.models["discriminator"](data_)

                loss = generator_criterion(fake_preds)
                val_total_loss += loss.item()

        avg_loss = val_total_loss / len(self.val_loader)

        return avg_loss

    def train_epoch(self, epoch, writer, log_interval):
        """训练单个epoch"""
        n_discriminator = self.cfg["training"].get("n_discriminator", 1)
        for _ in range(n_discriminator):
            d_loss = self.train_discriminator()
        g_loss = self.train_generator()

        return {"discriminator": d_loss, "generator": g_loss}

    def validate(self) -> dict[str, float]:
        """验证步骤"""
        d_loss = self.eval_discriminator()
        g_loss = self.eval_generator()
        return {"discriminator": d_loss, "generator": g_loss}  # 统一接口

    def eval(self, num_samples: int = 5) -> None:
        """可视化重构结果"""
        self.models["generator"].eval()
        self.models["discriminator"].eval()

        z = torch.randn((num_samples, self.z_dim), device=self.device, dtype=torch.float32)

        with torch.no_grad():
            recons = self.models["generator"](z)

        plot_figures(recons)


"""
Helper functions
"""


def discriminator_criterion(real_preds: torch.Tensor, fake_preds: torch.Tensor) -> torch.Tensor:
    real_loss = torch.nn.functional.binary_cross_entropy_with_logits(real_preds, torch.ones_like(real_preds))
    fake_loss = torch.nn.functional.binary_cross_entropy_with_logits(fake_preds, torch.zeros_like(fake_preds))
    return (real_loss + fake_loss) / 2.0


def generator_criterion(fake_preds: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.binary_cross_entropy_with_logits(fake_preds, torch.ones_like(fake_preds))
