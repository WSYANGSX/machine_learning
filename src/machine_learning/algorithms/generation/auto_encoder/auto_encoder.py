# 对于无监督式学习，比较好的办法是重建自己，通过重建数据发现数据的模态特征信息
# auto-encoder相当于对数据进行降维处理，类似PCA，只不过PCA是通过求解特征向量进行降维，是线性降维方式，而auto-encoder是非线性降维方式
from itertools import chain
from typing import Literal, Mapping

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from machine_learning.models import BaseNet
from machine_learning.algorithms.base import AlgorithmBase
from machine_learning.utils import plot_raw_recon_figures


class AutoEncoder(AlgorithmBase):
    def __init__(
        self,
        cfg: str,
        models: Mapping[str, BaseNet],
        name: str = "auto_encoder",
        device: Literal["cuda", "cpu", "auto"] = "auto",
    ) -> None:
        """
        自编码器实现

        parameters:
        - cfg (str): 配置文件路径(YAML格式).
        - models (Mapping[str, BaseNet]): auto-encoder算法所需模型.{"encoder":model1,"decoder":model2}.
        - name (str): 算法名称. Default to "auto_encoder".
        - device (str): 运行设备(auto自动选择).
        """
        super().__init__(cfg=cfg, models=models, name=name, device=device)

        # -------------------- 配置优化器 --------------------
        self._configure_optimizers()
        self._configure_schedulers()

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
            ValueError(f"暂时不支持优化器:{opt_cfg['type']}")

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
        """训练单个epoch"""
        self._models["encoder"].train()
        self._models["decoder"].train()

        total_loss = 0.0
        criterion = nn.MSELoss()

        for batch_idx, (data, _) in enumerate(self.train_loader):
            data = data.to(self._device, non_blocking=True)

            self._optimizers["ae"].zero_grad()

            z = self._models["encoder"](data)
            output = self._models["decoder"](z)

            loss = criterion(output, data)
            loss.backward()  # 反向传播计算各权重的梯度

            torch.nn.utils.clip_grad_norm_(self.params, self._cfg["optimizer"]["grad_clip"])
            self._optimizers["ae"].step()

            total_loss += loss.item()

            if batch_idx % log_interval == 0:
                writer.add_scalar(
                    "loss/train_batch", loss.item(), epoch * len(self.train_loader) + batch_idx
                )  # batch loss

        avg_loss = total_loss / len(self.train_loader)

        return {"ae": avg_loss}  # 统一接口

    def validate(self) -> dict[str, float]:
        """验证步骤"""
        self._models["encoder"].eval()
        self._models["decoder"].eval()

        total_loss = 0.0
        criterion = nn.MSELoss()

        with torch.no_grad():
            for data, _ in self.val_loader:
                data = data.to(self._device, non_blocking=True)
                z = self._models["encoder"](data)
                recon = self._models["decoder"](z)
                total_loss += criterion(recon, data).item()

        avg_loss = total_loss / len(self.val_loader)

        return {"ae": avg_loss, "save_metric": avg_loss}  # 统一接口

    def eval(self, num_samples: int = 5) -> None:
        """可视化重构结果"""
        self._models["encoder"].eval()
        self._models["decoder"].eval()

        data, _ = next(iter(self.val_loader))
        sample_indices = torch.randint(low=0, high=len(data), size=(num_samples,))
        data = data[sample_indices].to(self._device)

        with torch.no_grad():
            z = self._models["encoder"](data)
            recons = self._models["decoder"](z)

        plot_raw_recon_figures(data, recons)
