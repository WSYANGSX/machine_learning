from typing import Literal, Mapping, Any, Union
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from machine_learning.networks import BaseNet
from machine_learning.algorithms.base import AlgorithmBase
from machine_learning.types.aliases import FilePath
from machine_learning.utils.image import show_raw_and_recon_images


class AutoEncoder(AlgorithmBase):
    def __init__(
        self,
        cfg: FilePath | Mapping[str, Any],
        data: Mapping[str, Union[Dataset, Any]],
        net: BaseNet,
        name: str | None = "auto_encoder",
        device: Literal["cuda", "cpu", "auto"] = "auto",
    ) -> None:
        """
        Implementation of AutoEncoder algorithm

        Args:
            cfg (YamlFilePath, Mapping[str, Any]): Configuration of the algorithm, it can be yaml file path or cfg map.
            data (Mapping[str, Union[Dataset, Any]]): Parsed specific dataset data, must including train dataset and val
            dataset, may contain data information of the specific dataset.
            net (Mapping[str, BaseNet]): Neural neural required by the AutoEncoder algorithm.
            name (str, optional): Name of the algorithm. Defaults to "auto_encoder".
            device (Literal[&quot;cuda&quot;, &quot;cpu&quot;, &quot;auto&quot;], optional): Running device. Defaults to
            "auto"-automatic selection by algorithm.
        """
        super().__init__(cfg=cfg, data=data, net=net, name=name, device=device)

        # modifying the algorithm's metrics
        self.train_metrics = {"tloss": None}
        self.val_metrics = {"vloss": None, "sloss": None}

    def train_epoch(self, epoch: int, writer: SummaryWriter, log_interval: int = 10) -> dict[str, float]:
        """Train a single epoch"""
        self.set_train()

        tloss = None

        pbar = tqdm(enumerate(self.train_loader), total=self.num_batches)
        for batch_idx, (data, _) in pbar:
            batch_inters = epoch * self.num_batches + batch_idx

            data = data.to(self.device, non_blocking=True)

            with autocast(enabled=self.amp):  # Ensure that the autocast scope correctly covers the forward computation
                output = self.net(data)
                loss = self.criterion(output, data)

            self.backward(loss)
            self.optimizer_step(batch_inters)

            tloss = (tloss * batch_idx + loss.item()) / (batch_idx + 1) if tloss is not None else loss.item()
            self.train_metrics["tloss"] = tloss

            if batch_idx % log_interval == 0:
                writer.add_scalar("loss/train_batch", loss.item(), batch_inters)  # batch loss

            self.log_pbar("train", pbar, epoch, **self.train_metrics)

        return self.train_metrics

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Validate after a single train epoch"""
        self.set_eval()

        vloss = None

        print(("%10s" * (len(self.v_metrics) + 2)) % ("", "", *self.v_metrics))
        pbar = tqdm(enumerate(self.val_loader), total=len(self.val_loader))
        for batch_idx, (data, _) in pbar:
            data = data.to(self.device, non_blocking=True)
            recon = self.net(data)
            loss = self.criterion(recon, data)

            vloss = (vloss * batch_idx + loss.item()) / (batch_idx + 1) if vloss is not None else loss.item()

            # add value to val_metrics
            self.val_metrics["vloss"] = vloss
            self.val_metrics["sloss"] = vloss

            # log
            self.log_pbar("val", pbar, **self.val_metrics)

        return self.val_metrics

    def criterion(self, recon: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        fun = nn.MSELoss()
        return fun(recon, data)

    def eval(self, num_samples: int = 5) -> None:
        """Evaluate the model effect"""
        self.set_eval()

        data, _ = next(iter(self.test_loader)) if self.test_loader else next(iter(self.val_loader))
        sample_indices = torch.randint(low=0, high=len(data), size=(num_samples,))
        data = data[sample_indices].to(self._device)

        with torch.no_grad():
            recons = self.net(data)

        show_raw_and_recon_images(data, recons)
