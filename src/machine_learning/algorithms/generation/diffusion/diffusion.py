from typing import Literal, Mapping, Any, Union
from tqdm import trange

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from machine_learning.networks import BaseNet
from machine_learning.algorithms.base import AlgorithmBase
from machine_learning.types.aliases import FilePath
from machine_learning.utils.img import plot_imgs


class Diffusion(AlgorithmBase):
    def __init__(
        self,
        cfg: FilePath | Mapping[str, Any],
        models: Mapping[str, BaseNet],
        data: Mapping[str, Union[Dataset, Any]],
        name: str | None = "diffusion",
        device: Literal["cuda", "cpu", "auto"] = "auto",
    ):
        """
        Implementation of Diffusion algorithm

        Args:
            cfg (YamlFilePath, Mapping[str, Any]): Configuration of the algorithm, it can be yaml file path or cfg map.
            models (Mapping[str, BaseNet]): Models required by the Diffusion algorithm, {"noise_predictor": model}.
            data (Mapping[str, Union[Dataset, Any]]): Parsed specific dataset data, must including train dataset and val
            dataset, may contain data information of the specific dataset.
            name (str, optional): Name of the algorithm. Defaults to "diffusion".
            device (Literal[&quot;cuda&quot;, &quot;cpu&quot;, &quot;auto&quot;], optional): Running device. Defaults to
            "auto"-automatic selection by algorithm.
        """
        super().__init__(cfg=cfg, models=models, data=data, name=name, device=device)

        # main parameters of the algorithm
        self.time_steps = self.cfg["algorithm"].get("time_steps", 2000)

        # ------------------------ configure algo parameters -----------------------
        self._configure_factors(self.cfg["algorithm"]["beta"]["method"])

    def _configure_optimizers(self) -> None:
        opt_cfg = self.cfg["optimizer"]

        if opt_cfg["type"] == "Adam":
            self._optimizers.update(
                {
                    "noise_predictor": torch.optim.Adam(
                        params=self.models["noise_predictor"].parameters(),
                        lr=opt_cfg["learning_rate"],
                        betas=(opt_cfg["beta1"], opt_cfg["beta2"]),
                        eps=opt_cfg["eps"],
                        weight_decay=opt_cfg["weight_decay"],
                    )
                }
            )
        else:
            ValueError(f"Does not support optimizer:{opt_cfg['type']} currently.")

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

        # model factors
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
        """Train a single epoch"""
        self.set_train()

        total_loss = 0.0
        criterion = nn.MSELoss()

        for batch_idx, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device, non_blocking=True)
            if not hasattr(self, "data_shape"):
                self.data_shape = data.shape[1:]

            noise = torch.randn_like(data)
            time_step = torch.randint(1, self.time_steps + 1, (data.shape[0],), device=self.device)
            noisey_data_t = self.noisey_data_t(data, time_step, noise)

            noise_ = self._models["noise_predictor"](noisey_data_t, time_step)

            loss = criterion(noise, noise_)

            self._optimizers["noise_predictor"].zero_grad()
            loss.backward()  # Backpropagation is used to calculate the gradients of each weight.

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

        return {"noise_predictor loss": avg_loss}

    def validate(self) -> float:
        """Validate after a single train epoch"""
        self.set_eval()

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
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)

        return {"noise_predictor loss": avg_loss, "save": avg_loss}

    @torch.no_grad()
    def sample(self, data: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Infer the data of the previous moment based on the data of the current moment

        Args:
            data (torch.Tensor): The data at the current time step.
            t (torch.Tensor): The present time step.

        Returns:
            torch.Tensor: The predicted data of the previous time step.
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
        """Evaluate the model effect by visualizing reconstruction results"""
        self.set_eval()

        data = torch.randn(num_samples, *self.data_shape, device=self.device)

        print("[INFO] Start sampling...")

        epoch = 0
        for time_step in trange(self.time_steps, 0, -1, desc="Processing: "):
            time_step = torch.full((num_samples,), time_step, device=self.device)
            data = self.sample(data, time_step)
            epoch += 1

        plot_imgs(list(data), color_mode="gray")


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
    Extract the a value from the data corresponding to time t in the generated time series, then reshape/expand this
    scalar a value to match the same spatial dimensions as data (i.e., transform its shape from
    (batch_size, channels, 1, 1) to (batch_size, channels, height, width)) to enable broadcasting for subsequent
    operations.
    """
    batch_size = data_shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(data_shape) - 1)))
