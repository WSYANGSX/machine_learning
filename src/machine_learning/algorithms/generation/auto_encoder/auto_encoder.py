from typing import Literal, Mapping, Any

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from machine_learning.networks import BaseNet
from machine_learning.algorithms.base import AlgorithmBase
from machine_learning.types.aliases import FilePath
from machine_learning.utils.draw import show_raw_and_recon_images


class AutoEncoder(AlgorithmBase):
    def __init__(
        self,
        cfg: FilePath | Mapping[str, Any],
        net: BaseNet,
        name: str | None = "auto_encoder",
        device: Literal["cuda", "cpu", "auto"] = "auto",
    ) -> None:
        """
        Implementation of AutoEncoder algorithm

        Args:
            cfg (YamlFilePath, Mapping[str, Any]): Configuration of the algorithm, it can be yaml file path or cfg map.
            net (Mapping[str, BaseNet]): Neural neural required by the AutoEncoder algorithm.
            name (str, optional): Name of the algorithm. Defaults to "auto_encoder".
            device (Literal[&quot;cuda&quot;, &quot;cpu&quot;, &quot;auto&quot;], optional): Running device. Defaults to
            "auto"-automatic selection by algorithm.
        """
        super().__init__(cfg=cfg, net=net, name=name, device=device)

    def train_epoch(self, epoch: int, writer: SummaryWriter, log_interval: int = 10) -> dict[str, float]:
        """Train a single epoch"""
        self.set_train()

        total_loss = 0.0
        criterion = nn.MSELoss()

        for batch_idx, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            output = self.net(data)

            loss = criterion(output, data)
            loss.backward()  # Backpropagation is used to calculate the gradients of each weight.

            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self._cfg["optimizer"]["grad_clip"])
            self.optimizer.step()

            total_loss += loss.item()

            if batch_idx % log_interval == 0:
                writer.add_scalar(
                    "loss/train_batch", loss.item(), epoch * len(self.train_loader) + batch_idx
                )  # batch loss

        avg_loss = total_loss / len(self.train_loader)

        return {"loss": avg_loss}  # use dict to unify interface

    def validate(self) -> dict[str, float]:
        """Validate after a single train epoch"""
        self.set_eval()

        total_loss = 0.0
        criterion = nn.MSELoss()

        with torch.no_grad():
            for data, _ in self.val_loader:
                data = data.to(self.device, non_blocking=True)
                recon = self.net(data)
                total_loss += criterion(recon, data).item()

        avg_loss = total_loss / len(self.val_loader)

        return {"loss": avg_loss, "save metric": avg_loss}  # use dict to unify interface

    def eval(self, num_samples: int = 5) -> None:
        """Evaluate the model effect"""
        self.set_eval()

        data, _ = next(iter(self.test_loader)) if hasattr(self, "test_loader") else next(iter(self.val_loader))
        sample_indices = torch.randint(low=0, high=len(data), size=(num_samples,))
        data = data[sample_indices].to(self._device)

        with torch.no_grad():
            recons = self.net(data)

        show_raw_and_recon_images(data, recons)
