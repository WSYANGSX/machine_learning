from typing import Literal, Mapping, Any
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter

from machine_learning.networks import BaseNet
from machine_learning.algorithms.base import AlgorithmBase
from machine_learning.types.aliases import FilePath
from machine_learning.utils.image import plot_raw_and_recon_imgs, img_tensor2np


class AutoEncoder(AlgorithmBase):
    def __init__(
        self,
        cfg: FilePath | Mapping[str, Any],
        net: BaseNet | None = None,
        name: str | None = "auto_encoder",
        device: Literal["cuda", "cpu", "auto"] = "auto",
    ) -> None:
        """
        Implementation of AutoEncoder algorithm.

        Args:
            cfg (YamlFilePath, Mapping[str, Any]): Configuration of the algorithm, it can be yaml file path or cfg map.
            net (BaseNet): Neural neural required by the AutoEncoder algorithm.
            name (str, optional): Name of the algorithm. Defaults to "auto_encoder".
            device (Literal[&quot;cuda&quot;, &quot;cpu&quot;, &quot;auto&quot;], optional): Running device. Defaults to
            "auto"-automatic selection by algorithm.
        """
        super().__init__(cfg=cfg, net=net, name=name, device=device)

    def train_epoch(self, epoch: int, writer: SummaryWriter, log_interval: int = 10) -> dict[str, float]:
        """Train a single epoch"""
        super().train_epoch(epoch, writer, log_interval)

        # metrics
        metrics = {"tloss": None}
        self.print_metric_titles("train", metrics)

        tloss = None

        pbar = tqdm(enumerate(self.train_loader), total=self.train_batches)
        for i, sample in pbar:
            batch_inters = epoch * self.train_batches + i

            data = sample["data"].to(self.device, non_blocking=True)

            with autocast(
                device_type=str(self.device), enabled=self.amp
            ):  # Ensure that the autocast scope correctly covers the forward computation
                output = self.net(data)
                loss = self.criterion(output, data)

            self.backward(loss)
            self.optimizer_step(batch_inters)

            tloss = (tloss * i + loss.item()) / (i + 1) if tloss is not None else loss.item()
            metrics["tloss"] = tloss

            if i % log_interval == 0:
                writer.add_scalar("loss/train_batch", loss.item(), batch_inters)  # batch loss

            self.pbar_log("train", pbar, epoch, **metrics)

        return metrics

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Validate after a single train epoch"""
        super().validate()

        # metrics
        metrics = {"vloss": None, "sloss": None}
        self.print_metric_titles("val", metrics)

        vloss = None

        pbar = tqdm(enumerate(self.val_loader), total=self.val_batches)
        for i, sample in pbar:
            data = sample["data"].to(self.device, non_blocking=True)
            recon = self.net(data)
            loss = self.criterion(recon, data)

            vloss = (vloss * i + loss.item()) / (i + 1) if vloss is not None else loss.item()

            # add value to val_metrics
            metrics["vloss"] = vloss
            metrics["sloss"] = vloss

            # log
            self.pbar_log("val", pbar, **metrics)

        return metrics

    def criterion(self, recon: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        fun = nn.MSELoss()
        return fun(recon, data)

    def eval(self, num_samples: int = 5) -> None:
        """Evaluate the model effect"""
        self.set_eval()

        sample = next(iter(self.test_loader)) if self.test_loader else self.val_loader
        data = sample["data"]
        indices = torch.randint(low=0, high=len(data), size=(num_samples,))
        data = data[indices].to(self._device)

        with torch.no_grad():
            recons = self.net(data)

        plot_raw_and_recon_imgs(list(img_tensor2np(d) for d in data), list(img_tensor2np(r) for r in recons))
