from typing import Literal, Mapping, Any
from itertools import chain

from machine_learning.models import BaseNet
from machine_learning.algorithms.base import AlgorithmBase, YamlFilePath
from machine_learning.utils.draw import show_raw_and_recon_images

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class VAE(AlgorithmBase):
    def __init__(
        self,
        cfg: YamlFilePath | Mapping[str, Any],
        models: Mapping[str, BaseNet],
        name: str | None = "vae",
        device: Literal["cuda", "cpu", "auto"] = "auto",
    ) -> None:
        """
        Implementation of VAE algorithm

        Args:
            cfg (YamlFilePath, Mapping[str, Any]): Configuration of the algorithm, it can be yaml file path or cfg map.
            models (Mapping[str, BaseNet]): Models required by the VAE algorithm, {"encoder": model1,"decoder": model2}.
            name (str, optional): Name of the algorithm. Defaults to "vae".
            device (Literal[&quot;cuda&quot;, &quot;cpu&quot;, &quot;auto&quot;], optional): Running device. Defaults to "auto"-automatic selection by algorithm.
        """
        super().__init__(cfg, models, name, device)

    def _configure_optimizers(self) -> None:
        opt_cfg = self.cfg["optimizer"]

        self.params = chain(self.models["encoder"].parameters(), self.models["decoder"].parameters())

        if opt_cfg["type"] == "Adam":
            self._optimizers.update(
                {
                    "vae": torch.optim.Adam(
                        params=self.params,
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
        """train a single epoch"""
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
            loss.backward()

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
        """Validate after a single train epoch"""
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

                # When dealing with losses, follow the same loss handling method: either sum up the samples or average
                # the samples to keep the two at the same order of magnitude
                kl_d = 0.5 * torch.sum(mu.pow(2) + log_var.exp() - log_var - 1, dim=1)

                total_loss += (criterion(output, data) + kl_d.mean() * self.cfg["algorithm"]["beta"]).item()

        avg_loss = total_loss / len(self.val_loader)

        return {"vae": avg_loss, "save": avg_loss}

    def eval(self, num_samples: int = 5) -> None:
        """Evaluate the model effect by visualizing reconstruction results"""
        self.set_eval()

        data, _ = next(iter(self.val_loader))
        sample_indices = torch.randint(low=0, high=len(data), size=(num_samples,))
        data = data[sample_indices].to(self.device)

        with torch.no_grad():
            mu, log_var = self._models["encoder"](data)
            # std = torch.exp(0.5 * log_var)
            # z = mu + std * torch.randn_like(mu)
            recons = self._models["decoder"](mu)

        show_raw_and_recon_images(data, recons)
