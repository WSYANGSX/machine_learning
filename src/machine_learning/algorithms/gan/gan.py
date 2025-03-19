from typing import Literal, Mapping
from machine_learning.models import BaseNet
from machine_learning.algorithms.base import AlgorithmBase
from machine_learning.utils import plot_raw_recon_figures

import torch


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
        - models (Mapping[str, BaseNet]): vae算法所需模型.{"generator":model1,"discriminator":model2}.
        - name (str): 算法名称. Default to "gan".
        - device (str): 运行设备(auto自动选择).
        """
        super().__init__(cfg, models, name, device)

        # -------------------- 配置优化器 -------------------
        self._configure_optimizers()
        self._configure_schedulers()

        # --------------------- 先验 -----------------------
        self.prior = torch.distributions.MultivariateNormal(
            loc=torch.zeros(self.models["generator"].input_dim, device=self.device),
            covariance_matrix=torch.eye(self.models["generator"].input_dim, device=self.device),
        )

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

        if sched_config.get("type") == "ReduceLROnPlateau":
            self._schedulers.update(
                {
                    "generator": torch.optim.lr_scheduler.ReduceLROnPlateau(
                        self.models["generator"],
                        mode="min",
                        factor=sched_config.get("factor", 0.1),
                        patience=sched_config.get("patience", 10),
                    )
                }
            )
            self._schedulers.update(
                {
                    "discriminator": torch.optim.lr_scheduler.ReduceLROnPlateau(
                        self.models["discriminator"],
                        mode="min",
                        factor=sched_config.get("factor", 0.1),
                        patience=sched_config.get("patience", 10),
                    )
                }
            )

    def train_discriminator(self, epoch: int) -> float:
        """训练单个discriminator epoch"""
        self.models["generator"].eval()
        self.models["discriminator"].train()

        total_loss = 0.0

        for _, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device, non_blocking=True)

            z_prior = self.prior.sample((len(data),))
            data_ = self.generator(z_prior)
            self._optimizers["discriminator"].zero_grad()

            output_t = self.discriminator(data)
            output_f = self.discriminator(data_)

            loss = discriminator_criterion(output_t, output_f)
            loss.backward()  # 反向传播计算各权重的梯度

            torch.nn.utils.clip_grad_norm_(self.models["discriminator"].parameters(), self.cfg["training"]["grad_clip"])
            self._optimizers["discriminator"].step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)

        return {"d_loss": avg_loss}

    def eval_discriminator(self) -> float:
        self.models["discriminator"].eval()
        val_total_loss = 0.0

        with torch.no_grad():
            for data, _ in self.val_loader:
                data = data.to(self.device, non_blocking=True)

                z_prior = self.prior.sample((len(data),))
                data_ = self.generator(z_prior)

                output_t = self.discriminator(data)
                output_f = self.discriminator(data_)

                loss = discriminator_criterion(output_t, output_f)
                val_total_loss += loss.item()

        return val_total_loss / len(self.val_loader)

    def train_generator(self, epoch: int):
        """训练单个generator epoch"""
        self.models["generator"].train()
        self.models["discriminator"].eval()

        total_loss = 0.0

        for _, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device, non_blocking=True)

            z_prior = self.prior.sample((len(data),))
            data_ = self.generator(z_prior)

            self._optimizers["generator"].zero_grad()

            output_f = self.discriminator(data_)

            loss = generator_criterion(output_f)
            loss.backward()  # 反向传播计算各权重的梯度

            torch.nn.utils.clip_grad_norm_(self.models["generator"].parameters(), self.cfg["training"]["grad_clip"])
            self._optimizers["generator"].step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)

        return avg_loss

    def eval_generator(self) -> float:
        self.models["generator"].eval()
        val_total_loss = 0.0

        with torch.no_grad():
            for data, _ in self.val_loader:
                data = data.to(self.device, non_blocking=True)

                z_prior = self.prior.sample((len(data),))
                data_ = self.generator(z_prior)

                output_f = self.discriminator(data_)

                loss = generator_criterion(output_f)
                val_total_loss += loss.item()

        return val_total_loss / len(self.val_loader)

    def train_epoch(self, epoch, writer, log_interval):
        """训练单个epoch"""
        self._models["encoder"].train()
        self._models["decoder"].train()

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
            loss = criterion(output, data) + kl_d.mean() * self.cfg["training"]["beta"]
            loss.backward()  # 反向传播计算各权重的梯度

            torch.nn.utils.clip_grad_norm_(self.params, self.cfg["training"]["grad_clip"])
            self._optimizers["vae"].step()

            total_loss += loss.item()

            if batch_idx % log_interval == 0:
                writer.add_scalar("loss/train_batch", loss.item(), epoch * len(self.train_loader))  # batch loss
                writer.add_scalar("kl/train_batch", kl_d.mean().item(), epoch * len(self.train_loader))  # batch kl

        avg_loss = total_loss / len(self.train_loader)

        return {"vae": avg_loss}

    def eval(self, num_samples: int = 5) -> None:
        """可视化重构结果"""
        self._models["encoder"].eval()
        self._models["decoder"].eval()

        z = self.prior.sample((num_samples,))

        with torch.no_grad():
            recons = self.generator(z)

        plot_raw_recon_figures()


"""
Helper functions
"""


def discriminator_criterion(real_preds: torch.Tensor, fake_preds: torch.Tensor) -> float:
    real_loss = torch.nn.functional.binary_cross_entropy_with_logits(real_preds, torch.ones_like(real_preds))
    fake_loss = torch.nn.functional.binary_cross_entropy_with_logits(fake_preds, torch.zeros_like(fake_preds))
    return real_loss + fake_loss


def generator_criterion(fake_preds: torch.Tensor) -> float:
    return torch.nn.functional.binary_cross_entropy_with_logits(fake_preds, torch.ones_like(fake_preds))
