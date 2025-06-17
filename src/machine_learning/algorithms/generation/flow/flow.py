from typing import Literal, Mapping, Any
from tqdm import trange

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from machine_learning.models import BaseNet
from machine_learning.algorithms.base import AlgorithmBase, YamlFilePath
from machine_learning.utils.draw import show_image


class Flow(AlgorithmBase):
    def __init__(
        self,
        cfg: YamlFilePath | Mapping[str, Any],
        models: Mapping[str, BaseNet],
        name: str | None = "flow",
        device: Literal["cuda", "cpu", "auto"] = "auto",
    ):
        """
        Implementation of Flow algorithm

        Args:
            cfg (str, dict): Configuration of the algorithm, it can be yaml file path or cfg dict.
            models (dict[str, BaseNet]): Models required by the YOLOv3 algorithm, {"flow": model}.
            name (str): Name of the algorithm. Defaults to "flow".
            device (Literal[&quot;cuda&quot;, &quot;cpu&quot;, &quot;auto&quot;], optional): Running device. Defaults to "auto"-automatic selection by algorithm.
        """
        super().__init__(cfg, models, name, device)

    def _configure_optimizers(self) -> None:
        opt_cfg = self.cfg["optimizer"]

        if opt_cfg["type"] == "Adam":
            self._optimizers.update(
                {
                    "flow": torch.optim.Adam(
                        params=self.models["flow"].parameters(),
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
        sch_cfg = self.cfg["scheduler"]

        if sch_cfg and sch_cfg.get("type") == "ReduceLROnPlateau":
            self._schedulers.update(
                {
                    "flow": torch.optim.lr_scheduler.ReduceLROnPlateau(
                        self._optimizers["flow"],
                        mode="min",
                        factor=sch_cfg.get("factor", 0.1),
                        patience=sch_cfg.get("patience", 10),
                    )
                }
            )

    def train_epoch(self, epoch: int, writer: SummaryWriter, log_interval: int = 10) -> float:
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
            loss.backward()

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
