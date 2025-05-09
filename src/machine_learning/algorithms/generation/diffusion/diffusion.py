from typing import Literal, Mapping
from tqdm import trange

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from machine_learning.models import BaseNet
from machine_learning.algorithms.base import AlgorithmBase
from machine_learning.utils import plot_figures


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
        - cfg (str): 配置文件路径 (YAML格式).
        - models (Mapping[str, BaseNet]): diffusion算法所需模型.{"noise_predictor":model}.
        - name (str): 算法名称. Default to "diffusion".
        - device (str): 运行设备 (auto自动选择).
        """
        super().__init__(cfg, models, name, device)

        # -------------------- 配置优化器 --------------------
        self._configure_optimizers()
        self._configure_schedulers()

        # -------------------- 配置权重项 --------------------
        self.time_steps = self.cfg["algorithm"].get("time_steps", 2000)
        self._configure_factors(self.cfg["algorithm"]["beta"]["method"])

    def _configure_optimizers(self) -> None:
        opt_config = self.cfg["optimizer"]

        if opt_config["type"] == "Adam":
            self._optimizers.update(
                {
                    "noise_predictor": torch.optim.Adam(
                        params=self.models["noise_predictor"].parameters(),
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

        if sch_config and sch_config.get("type") == "ReduceLROnPlateau":
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
        start = self.cfg["algorithm"]["beta"].get("start", 0.0001)
        end = self.cfg["algorithm"]["beta"].get("end", 0.002)

        if method == "linear":
            self.betas = linear_beta_schedule(start, end, self.time_steps).to(self.device)
        elif method == "quadratic":
            self.betas = quadratic_beta_schedule(start, end, self.time_steps).to(self.device)
        elif method == "sigmoid":
            self.betas = sigmoid_beta_schedule(start, end, self.time_steps).to(self.device)
        else:
            raise ValueError(f"Method {method} to generate betas is not implemented.")

        # 模型因子
        self.alphas = 1 - self.betas  # index 0:T-1
        self.alphas_hat = torch.cumprod(self.alphas, dim=0)  # index 0:T-1
        self.alphas_hat_prev = torch.cat(
            [torch.ones(1, device=self.device), self.alphas_hat[:-1]], dim=0
        )  # 从x1倒推x0时，不再添加噪声，所以在首段添加1, index 0:T-1

        self.sqrt_alphas = torch.sqrt(self.alphas)  # index 0:T-1
        self.sqrt_alphas_hat = torch.sqrt(self.alphas_hat)  # index 0:T-1
        self.sqrt_one_minus_alphas_hat = torch.sqrt(1 - self.alphas_hat)  # index 0:T-1

        self.posterior_variance = self.betas * (1.0 - self.alphas_hat_prev) / (1.0 - self.alphas_hat)  # index 0:T-1

    def noisey_data_t(self, raw_data: torch.Tensor, time_step: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_hat, time_step - 1, raw_data.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_hat, time_step - 1, raw_data.shape)

        return sqrt_alphas_cumprod_t * raw_data + sqrt_one_minus_alphas_cumprod_t * noise

    def train_epoch(self, epoch: int, writer: SummaryWriter, log_interval: int = 10) -> float:
        """训练单个epoch"""
        self._models["noise_predictor"].train()

        total_loss = 0.0
        criterion = nn.MSELoss()

        for batch_idx, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device, non_blocking=True)
            noise = torch.randn_like(data)
            time_step = torch.randint(1, self.time_steps + 1, (data.shape[0],), device=self.device)
            noisey_data_t = self.noisey_data_t(data, time_step, noise)

            noise_ = self._models["noise_predictor"](noisey_data_t, time_step)

            loss = criterion(noise, noise_)

            self._optimizers["noise_predictor"].zero_grad()
            loss.backward()  # 反向传播计算各权重的梯度
            torch.nn.utils.clip_grad_norm_(
                self.models["noise_predictor"].parameters(), self.cfg["optimizer"]["grad_clip"]
            )
            self._optimizers["noise_predictor"].step()

            total_loss += loss.item()

            if batch_idx % log_interval == 0:
                writer.add_scalar(
                    "loss/train_batch", loss.item(), epoch * len(self.train_loader) + batch_idx
                )  # batch loss

        avg_loss = total_loss / len(self.train_loader)

        return {"noise_predictor": avg_loss}

    def validate(self) -> float:
        """验证步骤"""
        self._models["noise_predictor"].eval()

        total_loss = 0.0
        criterion = nn.MSELoss()

        with torch.no_grad():
            for data, _ in self.val_loader:
                data = data.to(self.device, non_blocking=True)
                noise = torch.randn_like(data)
                time_step = torch.randint(1, self.time_steps + 1, (data.shape[0],), device=self.device)
                noisey_data_t = self.noisey_data_t(data, time_step, noise)

                noise_ = self._models["noise_predictor"](noisey_data_t, time_step)

                loss = criterion(noise, noise_)
                total_loss += loss

        avg_loss = total_loss / len(self.val_loader)

        return {"noise_predictor": avg_loss, "save_metric": avg_loss}

    @torch.no_grad()
    def sample(self, data: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """基于当前时刻数据推断前一时刻的数据

        Args:
            data (torch.Tensor): 当前时刻输入的数据.
            t (torch.Tensor): 当前时刻.

        Returns:
            torch.Tensor: 前一时刻的预测数据.
        """
        beta_t = extract(self.betas, t - 1, data.shape)
        sqrt_one_minus_alphas_hat_t = extract(self.sqrt_one_minus_alphas_hat, t - 1, data.shape)
        sqrt_alphas_t = extract(self.sqrt_alphas, t - 1, data.shape)

        model_mean = (
            1 / sqrt_alphas_t * (data - beta_t * self.models["noise_predictor"](data, t) / sqrt_one_minus_alphas_hat_t)
        )

        if t[0] == 1:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t - 1, data.shape)
            noise = torch.randn_like(data)

            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def eval(self, num_samples: int = 5) -> None:
        """可视化重构结果"""
        self._models["noise_predictor"].eval()

        data = torch.randn(num_samples, *self.batch_data_shape[1:], device=self.device)

        print("[INFO] Start sampling...")

        epoch = 0
        for time_step in trange(self.time_steps, 0, -1, desc="Processing: "):
            time_step = torch.full((num_samples,), time_step, device=self.device)
            data = self.sample(data, time_step)
            epoch += 1

        plot_figures(data, cmap="gray")

    def _initialize_data_loader(self, train_data_loader, val_data_loader):
        super()._initialize_data_loader(train_data_loader, val_data_loader)

        data, labels = next(iter(self.train_loader))
        self.batch_data_shape = data.shape
        self.batch_label_shape = labels.shape
        print(
            "[INFO] Batch data shape: ", self.batch_data_shape, " " * 5, "Batch labels shape: ", self.batch_label_shape
        )


"""
Helper functions
"""


def quadratic_beta_schedule(start: float, end: float, time_steps: int) -> torch.Tensor:
    return torch.linspace(start**0.5, end**0.5, time_steps) ** 2


def linear_beta_schedule(start: float, end: float, time_steps: int) -> torch.Tensor:
    return torch.linspace(start, end, time_steps)


def sigmoid_beta_schedule(start: float, end: float, time_steps: int) -> torch.Tensor:
    betas = torch.linspace(-6, 6, time_steps)
    return torch.sigmoid(betas) * (end - start) + start


def extract(a: torch.Tensor, t: torch.Tensor, data_shape: tuple) -> torch.Tensor:
    """
    根据生成的时间序列 t 提取对应 data 的时间t时的a值, 然后将 a 值转换为与 data 相同的 size, 用于后续数据广播
    (batch_size, channels, 1, 1) -> (batch_size, channels, height, width)
    """
    batch_size = data_shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(data_shape) - 1)))
