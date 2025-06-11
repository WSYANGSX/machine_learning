from itertools import chain
from typing import Literal, Mapping
from machine_learning.models import BaseNet
from machine_learning.algorithms.base import AlgorithmBase
from machine_learning.utils import show_raw_and_recon_images
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class VQ_VAE(AlgorithmBase):
    def __init__(
        self,
        cfg: str,
        models: Mapping[str, BaseNet],
        name: str = "vq_vae",
        device: Literal["cuda", "cpu", "auto"] = "auto",
    ) -> None:
        """
        vq-vae算法实现

        parameters:
        - cfg (str): 配置文件路径(YAML格式).
        - models (Mapping[str, BaseNet]): vae算法所需模型. {"encoder":model1,"decoder":model2}.
        - name (str): 算法名称. Default to "vq_vae".
        - device (str): 运行设备(auto自动选择).
        """
        super().__init__(cfg, models, name, device)

        # ---------------- 配置embedding空间 -----------------
        self._configure_embedding()

        # -------------------- 配置优化器 --------------------
        self._configure_optimizers()
        self._configure_schedulers()

    def _configure_optimizers(self) -> None:
        opt_config = self.cfg["optimizer"]

        self.params = chain(
            self.models["encoder"].parameters(), self.models["decoder"].parameters(), self._embedding.parameters()
        )

        if opt_config["type"] == "Adam":
            self._optimizers.update(
                {
                    "vq_vae": torch.optim.Adam(
                        params=self.params,
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

        if sch_config.get("type") == "ReduceLROnPlateau":
            self._schedulers.update(
                {
                    "vq_vae": torch.optim.lr_scheduler.ReduceLROnPlateau(
                        self._optimizers["vq_vae"],
                        mode="min",
                        factor=sch_config.get("factor", 0.1),
                        patience=sch_config.get("patience", 10),
                    )
                }
            )

    def _configure_embedding(self) -> None:
        self._num_embeddings = self._cfg["model"]["num_embeddings"]
        self._embedding_dim = self._cfg["model"]["embedding_dim"]

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
        """训练单个epoch"""
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
            ).contiguous()  # 使梯度反向传播时quantized可以传到z上，进而传到encoder中
            output = self._models["decoder"](quantized)

            # loss
            recon_loss = criterion(output, data)
            commitment_loss = criterion(quantized.detach(), z)
            embedding_loss = criterion(quantized, z.detach())

            loss = recon_loss + commitment_loss + self.cfg["algorithm"]["beta"] * embedding_loss

            loss.backward()  # 反向传播计算各权重的梯度

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
        """验证步骤"""
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
        """可视化重构结果"""
        self.set_eval()

        data, _ = next(iter(self.val_loader))
        sample_indices = torch.randint(low=0, high=len(data), size=(num_samples,))
        data = data[sample_indices].to(self.device)

        z = self._models["encoder"](data)
        quantized = self.look_neighboring_vector(z)
        recons = self._models["decoder"](quantized)

        show_raw_and_recon_images(data, recons)
