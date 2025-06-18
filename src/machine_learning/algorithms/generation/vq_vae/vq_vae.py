from typing import Literal, Mapping, Any
from itertools import chain

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from machine_learning.models import BaseNet
from machine_learning.algorithms.base import AlgorithmBase
from machine_learning.types.aliases import FilePath
from machine_learning.utils.draw import show_raw_and_recon_images


class VQ_VAE(AlgorithmBase):
    def __init__(
        self,
        cfg: FilePath | Mapping[str, Any],
        models: Mapping[str, BaseNet],
        name: str | None = "vq_vae",
        device: Literal["cuda", "cpu", "auto"] = "auto",
    ) -> None:
        """
        Implementation of VQ_VAE algorithm

        Args:
            cfg (YamlFilePath, Mapping[str, Any]): Configuration of the algorithm, it can be yaml file path or cfg map.
            models (Mapping[str, BaseNet]): Models required by the VQ_VAE algorithm, {"encoder": model1,"decoder": model2}.
            name (str, optional): Name of the algorithm. Defaults to "vq_vae".
            device (Literal[&quot;cuda&quot;, &quot;cpu&quot;, &quot;auto&quot;], optional): Running device. Defaults to "auto"-automatic selection by algorithm.
        """
        super().__init__(cfg, models, name, device)

    def _configure_optimizers(self) -> None:
        opt_cfg = self.cfg["optimizer"]

        # ---------------- configure embedding space -----------------
        self._configure_embedding()

        self.params = chain(
            self.models["encoder"].parameters(), self.models["decoder"].parameters(), self._embedding.parameters()
        )

        if opt_cfg["type"] == "Adam":
            self._optimizers.update(
                {
                    "vq_vae": torch.optim.Adam(
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
        sch_cfg = self.cfg["scheduler"]

        if sch_cfg.get("type") == "ReduceLROnPlateau":
            self._schedulers.update(
                {
                    "vq_vae": torch.optim.lr_scheduler.ReduceLROnPlateau(
                        self._optimizers["vq_vae"],
                        mode="min",
                        factor=sch_cfg.get("factor", 0.1),
                        patience=sch_cfg.get("patience", 10),
                    )
                }
            )

    def _configure_embedding(self) -> None:
        self._num_embeddings = self.cfg["model"]["num_embeddings"]
        self._embedding_dim = self.cfg["model"]["embedding_dim"]

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim, device=self.device)
        self._embedding.weight.data.normal_(mean=0, std=0.08)

    def look_neighboring_vector(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        flatten_inputs = inputs.view(-1, self._embedding_dim)

        distances = torch.sum((torch.unsqueeze(flatten_inputs, dim=1) - self._embedding.weight) ** 2, dim=-1)

        encoding_indices = torch.argmin(distances, dim=-1)  # [0,1,5,5,...]
        quantized = self._embedding.weight[encoding_indices].contiguous()

        return quantized.view(inputs.shape).permute(0, 3, 1, 2)

    def train_epoch(self, epoch: int, writer: SummaryWriter, log_interval: int = 10) -> float:
        self.set_train()

        total_loss = 0.0
        criterion = nn.MSELoss()

        for batch_idx, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device, non_blocking=True)

            self._optimizers["vq_vae"].zero_grad()

            z = self._models["encoder"](data)  # [256, 64, 4, 4]
            quantized = self.look_neighboring_vector(z)
            quantized = (
                z + (quantized - z).detach()
            ).contiguous()  # When the gradient is backpropagated, the quantized can be passed to z and then to the encoder
            output = self._models["decoder"](quantized)

            # loss
            recon_loss = criterion(output, data)
            commitment_loss = criterion(quantized.detach(), z)
            embedding_loss = criterion(quantized, z.detach())

            loss = recon_loss + commitment_loss + self.cfg["algorithm"]["beta"] * embedding_loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.params, self.cfg["optimizer"]["grad_clip"])
            self._optimizers["vq_vae"].step()

            total_loss += loss.item()

            if batch_idx % log_interval == 0:
                writer.add_scalar(
                    "loss/train_batch", loss.item(), epoch * len(self.train_loader) + batch_idx
                )  # batch loss
                writer.add_scalar("loss_recon/", recon_loss.item(), epoch * len(self.train_loader) + batch_idx)
                writer.add_scalar("loss_commit/", commitment_loss.item(), epoch * len(self.train_loader) + batch_idx)
                writer.add_scalar("loss_embed/", embedding_loss.item(), epoch * len(self.train_loader) + batch_idx)

        avg_loss = total_loss / len(self.train_loader)

        return {"vq_vae": avg_loss}

    @torch.no_grad()
    def validate(self) -> float:
        self.set_eval()

        total_loss = 0.0
        criterion = nn.MSELoss()

        for data, _ in self.val_loader:
            data = data.to(self.device, non_blocking=True)

            z = self._models["encoder"](data)
            quantized = self.look_neighboring_vector(z)
            output = self._models["decoder"](quantized)

            total_loss += criterion(output, data).item()

        avg_loss = total_loss / len(self.val_loader)

        return {"vq_vae": avg_loss, "save": avg_loss}

    @torch.no_grad()
    def eval(self, num_samples: int = 5) -> None:
        self.set_eval()

        data, _ = next(iter(self.val_loader))
        sample_indices = torch.randint(low=0, high=len(data), size=(num_samples,))
        data = data[sample_indices].to(self.device)

        z = self._models["encoder"](data)
        quantized = self.look_neighboring_vector(z)
        recons = self._models["decoder"](quantized)

        show_raw_and_recon_images(data, recons)
