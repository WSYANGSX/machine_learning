from typing import Literal, Mapping
from tqdm import trange

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from machine_learning.models import BaseNet
from machine_learning.algorithms.base import AlgorithmBase
from machine_learning.utils import show_image


class Flow(AlgorithmBase):
    def __init__(
        self,
        cfg: str,
        models: Mapping[str, BaseNet],
        name: str = "diffusion",
        device: Literal["cuda", "cpu", "auto"] = "auto",
    ):
        """
        流模型实现

        parameters:
        - cfg (str): 配置文件路径 (YAML格式).
        - models (Mapping[str, BaseNet]): flow算法所需模型.{"flow": model}.
        - name (str): 算法名称. Default to "flow".
        - device (str): 运行设备 (auto自动选择).
        """
        super().__init__(cfg, models, name, device)

        # -------------------- 配置优化器 --------------------
        self._configure_optimizers()
        self._configure_schedulers()

    def _configure_optimizers(self) -> None:
        opt_config = self.cfg["optimizer"]

        if opt_config["type"] == "Adam":
            self._optimizers.update(
                {
                    "flow": torch.optim.Adam(
                        params=self.models["flow"].parameters(),
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
                    "flow": torch.optim.lr_scheduler.ReduceLROnPlateau(
                        self._optimizers["flow"],
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
            noise = torch.randn_like(data)
            time_step = torch.randint(1, self.time_steps + 1, (data.shape[0],), device=self.device)
            noisey_data_t = self.noisey_data_t(data, time_step, noise)

            noise_ = self._models["flow"](noisey_data_t, time_step)

            loss = criterion(noise, noise_)

            self._optimizers["noise_predictor"].zero_grad()
            loss.backward()  # 反向传播计算各权重的梯度
            torch.nn.utils.clip_grad_norm_(
                self.models["noise_predictor"].parameters(), self.cfg["training"]["grad_clip"]
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
                total_loss += loss

        avg_loss = total_loss / len(self.val_loader)

        return {"noise_predictor": avg_loss, "save_metric": avg_loss}

    @torch.no_grad()
    def eval(self, num_samples: int = 5) -> None:
        """可视化重构结果"""
        self.set_eval()

        data = torch.randn(num_samples, *self.batch_data_shape[1:], device=self.device)

        print("[INFO] Start sampling...")

        epoch = 0
        for time_step in trange(self.time_steps, 0, -1, desc="Processing: "):
            time_step = torch.full((num_samples,), time_step, device=self.device)
            data = self.sample(data, time_step)
            epoch += 1

        show_image(data, color_mode="gray")


"""
Helper functions
"""
