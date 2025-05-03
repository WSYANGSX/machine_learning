from typing import Literal, Mapping

import torch
from torch.utils.tensorboard import SummaryWriter

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
        sched_config = self.cfg.get("scheduler", {})

        if sched_config is not None and sched_config.get("type") == "StepLR":
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

    def train_epoch(self, epoch: int, writer: SummaryWriter, log_interval: int = 10):
        """训练单个epoch"""
        total_d_loss = 0.0
        total_g_loss = 0.0
        g_count = 0

        n_discriminator = self.cfg["training"].get("n_discriminator", 1)

        for batch_idx, (real_images, _) in enumerate(self.train_loader):
            real_images = real_images.to(self._device, non_blocking=True)
            z = torch.randn((len(real_images), self.z_dim), device=self.device, dtype=torch.float32)
            fake_image = self.models["generator"](z)

            # 训练 discriminator
            self._optimizers["discriminator"].zero_grad()

            real_preds = self.models["discriminator"](real_images)
            fake_preds = self.models["discriminator"](fake_image.detach())

            d_loss = discriminator_criterion_bce(real_preds, fake_preds)
            d_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.models["discriminator"].parameters(), self.cfg["training"]["grad_clip"]["discriminator"]
            )
            self._optimizers["discriminator"].step()
            total_d_loss += d_loss

            real_accuracy = (real_preds > 0.5).float().mean()
            fake_accuracy = (fake_preds < 0.5).float().mean()

            # 训练 generator
            if batch_idx % n_discriminator == 0:
                self._optimizers["generator"].zero_grad()

                fake_preds = self.models["discriminator"](fake_image)
                g_loss = generator_criterion_bce(fake_preds)
                total_g_loss += g_loss

                g_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.models["generator"].parameters(), self.cfg["training"]["grad_clip"]["generator"]
                )
                self._optimizers["generator"].step()

                g_count += 1

            # 记录数据
            if batch_idx % log_interval == 0:
                writer.add_scalar("d_loss/train_batch", d_loss.item(), epoch * len(self.train_loader) + batch_idx)
                writer.add_scalar("g_loss/train_batch", g_loss.item(), epoch * len(self.train_loader) + batch_idx)
                writer.add_scalar(
                    "real_accuracy/train_batch", real_accuracy.item(), epoch * len(self.train_loader) + batch_idx
                )
                writer.add_scalar(
                    "fake_accuracy/train_batch", fake_accuracy.item(), epoch * len(self.train_loader) + batch_idx
                )

        d_avg_loss = total_d_loss / len(self.train_loader)
        g_avg_loss = total_g_loss / g_count

        return {"discriminator": d_avg_loss, "generator": g_avg_loss}

    def validate(self) -> dict[str, float]:
        """验证步骤"""
        self.models["generator"].eval()
        self.models["discriminator"].eval()

        d_total_loss = 0.0
        g_total_loss = 0.0

        with torch.no_grad():
            for real_image, _ in self.val_loader:
                real_image = real_image.to(self.device, non_blocking=True)
                z = torch.randn((len(real_image), self.z_dim), device=self.device, dtype=torch.float32)
                fake_image = self.models["generator"](z)

                real_preds = self.models["discriminator"](real_image)
                fake_preds = self.models["discriminator"](fake_image)

                d_loss = discriminator_criterion_bce(real_preds, fake_preds)
                d_total_loss += d_loss.item()

                g_loss = generator_criterion(fake_preds)
                g_total_loss += g_loss.item()

        d_avg_loss = d_total_loss / len(self.val_loader)
        g_avg_loss = g_total_loss / len(self.val_loader)

        return {"discriminator": d_avg_loss, "generator": g_avg_loss}  # 统一接口

    def eval(self, num_samples: int = 5) -> None:
        """可视化重构结果"""
        self.models["generator"].eval()  # 将模型切换到评估模式，主要是影响层的行为，比如dropout层停止随机丢弃神经元
        self.models["discriminator"].eval()

        z = torch.randn((num_samples, self.z_dim), device=self.device, dtype=torch.float32)

        with torch.no_grad():  # 禁用梯度计算，作用与.detach()相同
            recons = self.models["generator"](z)

        plot_figures(recons, cmap="gray")


"""
Helper functions
"""


# 方案1 loss函数
def discriminator_criterion(real_preds: torch.Tensor, fake_preds: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    loss = -(torch.log(real_preds + eps).mean() + torch.log(1 - fake_preds + eps).mean())
    return loss


def generator_criterion(fake_preds: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    loss = -torch.log(fake_preds + eps).mean()
    return loss


# 方案2 loss函数
def discriminator_criterion_bce(real_preds: torch.Tensor, fake_preds: torch.Tensor) -> torch.Tensor:
    real_loss = torch.nn.functional.binary_cross_entropy(real_preds, torch.ones_like(real_preds))
    fake_loss = torch.nn.functional.binary_cross_entropy(fake_preds, torch.zeros_like(fake_preds))
    return real_loss + fake_loss


def generator_criterion_bce(fake_preds: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.binary_cross_entropy(fake_preds, torch.ones_like(fake_preds))


# 方案3 loss函数
def discriminator_criterion_bce_logits(real_preds: torch.Tensor, fake_preds: torch.Tensor) -> torch.Tensor:
    real_loss = torch.nn.functional.binary_cross_entropy_with_logits(real_preds, torch.ones_like(real_preds))
    fake_loss = torch.nn.functional.binary_cross_entropy_with_logits(fake_preds, torch.zeros_like(fake_preds))
    return real_loss + fake_loss


def generator_criterion_bce_logits(fake_preds: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.binary_cross_entropy_with_logits(fake_preds, torch.ones_like(fake_preds))
