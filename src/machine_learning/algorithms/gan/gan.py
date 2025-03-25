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

    def train_epoch(self, epoch, writer, log_interval):
        """训练单个epoch"""
        total_d_loss = 0.0
        total_g_loss = 0.0

        for batch_idex, (real_images, _) in enumerate(self.train_loader):
            real_images = real_images.to(self._device, non_blocking=True)
            z = torch.randn((len(real_images), self.z_dim), device=self.device, dtype=torch.float32)
            fake_image = self.models["generator"](z)

            # 训练 discriminator
            n_discriminator = self.cfg["training"].get("n_discriminator", 1)
            for _ in range(n_discriminator):
                self._optimizers["discriminator"].zero_grad()

                real_preds = self.models["discriminator"](real_images)
                fake_preds = self.models["discriminator"](fake_image.detach())

                d_loss = discriminator_criterion(real_preds, fake_preds)
                d_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.models["discriminator"].parameters(), self.cfg["training"]["grad_clip"]["discriminator"]
                )
                self._optimizers["discriminator"].step()
            total_d_loss += d_loss

            # 训练 generator
            self._optimizers["generator"].zero_grad()
            fake_preds = self.models["discriminator"](fake_image)
            g_loss = generator_criterion(fake_preds)
            total_g_loss += g_loss
            g_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.models["generator"].parameters(), self.cfg["training"]["grad_clip"]["generator"]
            )
            self._optimizers["generator"].step()

        d_avg_loss = d_loss / len(self.train_loader)
        g_avg_loss = g_loss / len(self.train_loader)

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

                d_loss = discriminator_criterion(real_preds, fake_preds)
                d_total_loss += d_loss.item()

                g_loss = generator_criterion(fake_preds)
                g_total_loss += g_loss.item()

        d_avg_loss = d_total_loss / len(self.val_loader)
        g_avg_loss = g_total_loss / len(self.val_loader)

        return {"discriminator": d_avg_loss, "generator": g_avg_loss}  # 统一接口

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
