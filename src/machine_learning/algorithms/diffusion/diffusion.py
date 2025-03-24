from typing import Literal, Mapping
from machine_learning.models import BaseNet
from machine_learning.algorithms.base import AlgorithmBase
from machine_learning.utils import plot_raw_recon_figures

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class Diffusion(AlgorithmBase):
    def __init__(
        self,
        cfg: str,
        models: Mapping[str, BaseNet],
        name: str = "diffusion",
        device: Literal["cuda", "cpu", "auto"] = "auto",
    ):
        """
        扩散模型实现

        parameters:
        - cfg (str): 配置文件路径(YAML格式).
        - models (Mapping[str, BaseNet]): diffusion算法所需模型.{"noisemaker":model}.
        - name (str): 算法名称. Default to "diffusion".
        - device (str): 运行设备(auto自动选择).
        """
        super().__init__(cfg, models, name, device)

        # -------------------- 配置优化器 --------------------
        self._configure_optimizers()
        self._configure_schedulers()

        # -------------------- 配置权重项 --------------------
        self._configure_factors(self.cfg["training"]["beta"]["method"])

        self.time_steps = self.cfg["training"].get("time_steps", 2000)

    def _configure_optimizers(self) -> None:
        opt_config = self.cfg["optimizer"]

        self.params = self.models["noise_predictor"].parameters()

        if opt_config["type"] == "Adam":
            self._optimizers.update(
                {
                    "noise_predictor": torch.optim.Adam(
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
                    "noise_predictor": torch.optim.lr_scheduler.ReduceLROnPlateau(
                        self._optimizers["noise_predictor"],
                        mode="min",
                        factor=sch_config.get("factor", 0.1),
                        patience=sch_config.get("patience", 10),
                    )
                }
            )

    def _configure_factors(self, method: str) -> None:
        start = self.cfg["training"]["beta"].get("start", 0.0001)
        end = self.cfg["training"]["beta"].get("end", 0.002)

        if method == "linear":
            self.betas = linear_beta_schedule(start, end, self.time_steps)
        elif method == "quadratic":
            self.betas = quadratic_beta_schedule(start, end, self.time_steps)
        elif method == "sigmoid":
            self.betas = sigmoid_beta_schedule(start, end, self.time_steps)
        else:
            raise ValueError(f"Method {method} to generate betas is not implemented.")

        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]], dim=0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def noisey_data_t(self, data: torch.Tensor, time_step: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(data)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, time_step, data.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, time_step, data.shape)

        return sqrt_alphas_cumprod_t * data + sqrt_one_minus_alphas_cumprod_t * noise

    def train_epoch(self, epoch: int, writer: SummaryWriter, log_interval: int = 10) -> float:
        """训练单个epoch"""
        self._models["noise_predictor"].train()

        total_loss = 0.0
        criterion = nn.MSELoss()

        for batch_idx, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device, non_blocking=True)
            noise = torch.randn_like(data)
            time_step = torch.randint(1, self.time_steps)
            noisey_data_t = self.noisey_data_t(data, time_step)

            self._optimizers["noise_predictor"].zero_grad()

            noise = self._models["noise_predictor"](noisey_data_t, time_step)

            loss = criterion(noise, noisey_data_t)
            loss.backward()  # 反向传播计算各权重的梯度

            torch.nn.utils.clip_grad_norm_(self.params, self.cfg["training"]["grad_clip"])
            self._optimizers["noise_predictor"].step()

            total_loss += loss.item()

            if batch_idx % log_interval == 0:
                writer.add_scalar("loss/train_batch", loss.item(), epoch * len(self.train_loader))  # batch loss

        avg_loss = total_loss / len(self.train_loader)

        return {"noise_predictor": avg_loss}

    def validate(self) -> float:
        """验证步骤"""
        self._models["encoder"].eval()
        self._models["decoder"].eval()

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

                total_loss += (criterion(output, data) + kl_d.mean() * self.cfg["training"]["beta"]).item()

        avg_loss = total_loss / len(self.val_loader)

        return {"vae": avg_loss, "save_metric": avg_loss}

    def eval(self, num_samples: int = 5) -> None:
        """可视化重构结果"""
        self._models["encoder"].eval()
        self._models["decoder"].eval()

        data, _ = next(iter(self.val_loader))
        sample_indices = torch.randint(low=0, high=len(data), size=(num_samples,))
        data = data[sample_indices].to(self.device)

        with torch.no_grad():
            mu, log_var = self._models["encoder"](data)
            # std = torch.exp(0.5 * log_var)
            # z = mu + std * torch.randn_like(mu)
            recons = self._models["decoder"](mu)

        plot_raw_recon_figures(data, recons)


"""
Helper functions
"""


def quadratic_beta_schedule(start: float, end: float, time_steps: int) -> torch.Tensor:
    return torch.linspace(start**0.5, end**0.5, time_steps) ** 2


def linear_beta_schedule(start: float, end: float, time_steps: int) -> torch.Tensor:
    return torch.linspace(start, end, time_steps)


def sigmoid_beta_schedule(start: float, end: float, time_steps: int) -> torch.Tensor:
    betas = torch.linspace(-6, 6, time_steps)
    return torch.sigmoid(betas) * (end - start) * start


def extract(a: torch.Tensor, t: torch.Tensor, data_shape: tuple) -> torch.Tensor:
    """
    根据生成的时间序列 t 提取对应 data 的时间t时的a值, 然后将a值转换为与 data 相同的size, 用于后续数据广播
    (batch_size,channels,1,1)->(batch_size,channels,height,width)
    """

    batch_size = data_shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(data_shape) - 1)))
