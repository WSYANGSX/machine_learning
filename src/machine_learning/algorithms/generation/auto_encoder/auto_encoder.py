from typing import Literal, Mapping, Any
from itertools import chain

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from machine_learning.models import BaseNet
from machine_learning.algorithms.base import AlgorithmBase, YamlFilePath
from machine_learning.utils.draw import show_raw_and_recon_images


class AutoEncoder(AlgorithmBase):
    def __init__(
        self,
        cfg: YamlFilePath | Mapping[str, Any],
        models: Mapping[str, BaseNet],
        name: str | None = "auto_encoder",
        device: Literal["cuda", "cpu", "auto"] = "auto",
    ) -> None:
        """
        Implementation of AutoEncoder algorithm

        Args:
            cfg (YamlFilePath, Mapping[str, Any]): Configuration of the algorithm, it can be yaml file path or cfg map.
            models (Mapping[str, BaseNet]): Models required by the AutoEncoder algorithm, {"encoder": model1, "decoder": model2}.
            name (str, optional): Name of the algorithm. Defaults to "auto_encoder".
            device (Literal[&quot;cuda&quot;, &quot;cpu&quot;, &quot;auto&quot;], optional): Running device. Defaults to "auto"-automatic selection by algorithm.
        """
        super().__init__(cfg=cfg, models=models, name=name, device=device)

    def _configure_optimizers(self) -> None:
        opt_cfg = self._cfg["optimizer"]

        self.params = chain(self.models["encoder"].parameters(), self.models["decoder"].parameters())

        if opt_cfg["type"] == "Adam":
            self._optimizers.update(
                {
                    "ae": torch.optim.Adam(
                        params=self.params,
                        lr=opt_cfg["learning_rate"],
                        betas=(opt_cfg["beta1"], opt_cfg["beta2"]),
                        eps=opt_cfg["eps"],
                        weight_decay=opt_cfg["weight_decay"],
                    ),
                }
            )
        else:
            ValueError(f"Does not support optimizer:{opt_cfg['type']} currently.")

    def _configure_schedulers(self) -> None:
        sch_config = self._cfg["scheduler"]

        if sch_config.get("type") == "ReduceLROnPlateau":
            self._schedulers.update(
                {
                    "ae": torch.optim.lr_scheduler.ReduceLROnPlateau(
                        self._optimizers["ae"],
                        mode="min",
                        factor=sch_config.get("factor", 0.1),
                        patience=sch_config.get("patience", 10),
                    )
                }
            )

    def train_epoch(self, epoch: int, writer: SummaryWriter, log_interval: int = 10) -> dict[str, float]:
        """Train a single epoch"""
        self.set_train()

        total_loss = 0.0
        criterion = nn.MSELoss()

        for batch_idx, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device, non_blocking=True)

            self._optimizers["ae"].zero_grad()

            z = self._models["encoder"](data)
            output = self._models["decoder"](z)

            loss = criterion(output, data)
            loss.backward()  # Backpropagation is used to calculate the gradients of each weight.

            torch.nn.utils.clip_grad_norm_(self.params, self._cfg["optimizer"]["grad_clip"])
            self._optimizers["ae"].step()

            total_loss += loss.item()

            if batch_idx % log_interval == 0:
                writer.add_scalar(
                    "loss/train_batch", loss.item(), epoch * len(self.train_loader) + batch_idx
                )  # batch loss

        avg_loss = total_loss / len(self.train_loader)

        return {"ae": avg_loss}  # use dict to unify interface

    def validate(self) -> dict[str, float]:
        """Validate after a single train epoch"""
        self.set_eval()

        total_loss = 0.0
        criterion = nn.MSELoss()

        with torch.no_grad():
            for data, _ in self.val_loader:
                data = data.to(self.device, non_blocking=True)
                z = self._models["encoder"](data)
                recon = self._models["decoder"](z)
                total_loss += criterion(recon, data).item()

        avg_loss = total_loss / len(self.val_loader)

        return {"ae": avg_loss, "save": avg_loss}  # use dict to unify interface

    def eval(self, num_samples: int = 5) -> None:
        """Evaluate the model effect"""
        self.set_eval()

        data, _ = next(iter(self.val_loader))
        sample_indices = torch.randint(low=0, high=len(data), size=(num_samples,))
        data = data[sample_indices].to(self._device)

        with torch.no_grad():
            z = self._models["encoder"](data)
            recons = self._models["decoder"](z)

        show_raw_and_recon_images(data, recons)
