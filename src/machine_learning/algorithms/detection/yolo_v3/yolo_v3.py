from typing import Literal

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from machine_learning.models import BaseNet
from machine_learning.algorithms.base import AlgorithmBase


class YoloV3(AlgorithmBase):
    def __init__(
        self,
        cfg: str,
        models: dict[str, BaseNet],
        name: str = "yolo_v3",
        device: Literal["cuda", "cpu", "auto"] = "auto",
    ) -> None:
        """
        Yolov3目标检测器算法实现

        parameters:
        - cfg (str): 配置文件路径(YAML格式).
        - models (Mapping[str, BaseNet]): Yolov3算法所需模型.{"darknet":model1}.
        - name (str): 算法名称. Default to "yolo_v3".
        - device (str): 运行设备(auto自动选择).
        """
        super().__init__(cfg=cfg, models=models, name=name, device=device)

        # -------------------- 配置优化器 --------------------
        self._configure_optimizers()
        self._configure_schedulers()

    def _configure_optimizers(self) -> None:
        opt_cfg = self._cfg["optimizer"]

        self.params = self.models["darknet"].parameters()

        if opt_cfg["type"] == "Adam":
            self._optimizers.update(
                {
                    "darknet": torch.optim.Adam(
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
                    "darknet": torch.optim.lr_scheduler.ReduceLROnPlateau(
                        self._optimizers["darknet"],
                        mode="min",
                        factor=sch_config.get("factor", 0.1),
                        patience=sch_config.get("patience", 10),
                    )
                }
            )

    def train_epoch(self, epoch: int, writer: SummaryWriter, log_interval: int = 10) -> dict[str, float]:
        """训练单个epoch"""
        self._models["darknet"].train()

        total_loss = 0.0
        criterion = nn.MSELoss()

        for batch_idx, (data, _) in enumerate(self.train_loader):
            data = data.to(self._device, non_blocking=True)

            self._optimizers["darknet"].zero_grad()

            z = self._models["darknet"](data)

            loss = criterion(output, data)
            loss.backward()  # 反向传播计算各权重的梯度

            torch.nn.utils.clip_grad_norm_(self.params, self._cfg["training"]["grad_clip"])
            self._optimizers["darknet"].step()

            total_loss += loss.item()

            if batch_idx % log_interval == 0:
                writer.add_scalar(
                    "loss/train_batch", loss.item(), epoch * len(self.train_loader) + batch_idx
                )  # batch loss

        avg_loss = total_loss / len(self.train_loader)

        return {"darknet": avg_loss}  # 统一接口

    def validate(self) -> dict[str, float]:
        """验证步骤"""
        self._models["darknet"].eval()

        total_loss = 0.0
        criterion = nn.MSELoss()

        with torch.no_grad():
            for data, _ in self.val_loader:
                data = data.to(self._device, non_blocking=True)
                z = self._models["darknet"](data)
                total_loss += criterion(recon, data).item()

        avg_loss = total_loss / len(self.val_loader)

        return {"darknet": avg_loss, "save_metric": avg_loss}  # 统一接口

    def eval(self, num_samples: int = 5) -> None:
        """可视化重构结果"""
        self._models["darknet"].eval()
