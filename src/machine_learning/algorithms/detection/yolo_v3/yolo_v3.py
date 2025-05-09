from typing import Literal
from itertools import chain

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from machine_learning.models import BaseNet
from machine_learning.algorithms.base import AlgorithmBase


class YoloV3(AlgorithmBase):
    def __init__(
        self,
        cfg: str | dict,
        models: dict[str, BaseNet],
        name: str = "yolo_v3",
        device: Literal["cuda", "cpu", "auto"] = "auto",
    ) -> None:
        """
        Yolov3目标检测器算法实现

        parameters:
        - cfg (str): 配置文件路径(YAML格式).
        - models (Mapping[str, BaseNet]): Yolov3算法所需模型, {"darknet":model1, "fpn":model2}.
        - name (str): 算法名称. Default to "yolo_v3".
        - device (str): 运行设备(auto自动选择).
        """
        super().__init__(cfg=cfg, models=models, name=name, device=device)

        # 配置主要参数
        self.num_classes = self._cfg["algorithm"]["num_classes"]
        self.num_anchors = self._cfg["algorithm"]["num_anchors"]
        self.anchor_sizes = self._cfg["algorithm"]["anchor_sizes"]
        self.image_size = self._cfg["algorithm"]["image_size"]

        # -------------------- 配置优化器 --------------------
        self._configure_optimizers()
        self._configure_schedulers()

    def _configure_optimizers(self) -> None:
        opt_cfg = self._cfg["optimizer"]

        self.params = chain([self.models["darknet"].parameters(), self.models["fpn"]])

        if opt_cfg["type"] == "Adam":
            self._optimizers.update(
                {
                    "yolo": torch.optim.Adam(
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
                    "yolo": torch.optim.lr_scheduler.ReduceLROnPlateau(
                        self._optimizers["yolo"],
                        mode="min",
                        factor=sch_config.get("factor", 0.1),
                        patience=sch_config.get("patience", 10),
                    )
                }
            )

    def train_epoch(self, epoch: int, writer: SummaryWriter, log_interval: int = 10) -> dict[str, float]:
        """训练单个epoch"""
        self.models["darknet"].train()
        self.models["fpn"].train()

        total_loss = 0.0

        for batch_idx, (data, labels) in enumerate(self.train_loader):
            data = data.to(self._device, non_blocking=True)

            self._optimizers["yolo"].zero_grad()

            skips = self.models["darknet"](data)
            fimg1, fimg2, fimg3 = self.models["fpn"](skips)

            loss = self.criterion(fimg1, fimg2, fimg3, labels)
            loss.backward()  # 反向传播计算各权重的梯度

            torch.nn.utils.clip_grad_norm_(self.params, self._cfg["optimizer"]["grad_clip"])
            self._optimizers["yolo"].step()

            total_loss += loss.item()

            if batch_idx % log_interval == 0:
                writer.add_scalar(
                    "loss/train_batch", loss.item(), epoch * len(self.train_loader) + batch_idx
                )  # batch loss

        avg_loss = total_loss / len(self.train_loader)

        return {"yolo": avg_loss}

    def validate(self) -> dict[str, float]:
        """验证步骤"""
        self.models["darknet"].eval()
        self.models["fpn"].eval()

        total_loss = 0.0
        criterion = nn.MSELoss()

        with torch.no_grad():
            for data, _ in self.val_loader:
                data = data.to(self._device, non_blocking=True)
                skips = self.models["darknet"](data)
                det1, det2, det3 = self.models["fpn"](skips)
                total_loss += criterion(det1, det2, det3, data).item()

        avg_loss = total_loss / len(self.val_loader)

        return {"yolo": avg_loss, "save_metric": avg_loss}

    def eval(self, num_samples: int = 5) -> None:
        pass

    # 损失函数设计
    def criterion(
        self,
        feature_image1: torch.Tensor,
        feature_image2: torch.Tensor,
        feature_image3: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        prediction1 = self.feature_decode(feature_image1)
        prediction2 = self.feature_decode(feature_image2)
        prediction3 = self.feature_decode(feature_image3)

        return prediction1, prediction2, prediction3

    # 特征图解码
    @torch.jit.script
    def feature_decode(self, feature_image: torch.Tensor) -> torch.Tensor:
        # 调整维度顺序 [B,C,H,W] -> [B,H,W,C]
        prediction = feature_image.permute(0, 2, 3, 1).contiguous()
        B, H, W, _ = prediction.shape
        stride = self.image_size // H  # 计算步长

        # 根据特征图尺寸选择锚框
        if H == 52:
            anchor_sizes = self.anchor_sizes[:3] / stride
        elif H == 26:
            anchor_sizes = self.anchor_sizes[3:6] / stride
        else:
            anchor_sizes = self.anchor_sizes[6:9] / stride

        # 构建偏移矩阵
        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        grid_xy = torch.stack((grid_x, grid_y), dim=-1).to(prediction.device)  # [H, W, 2]

        # 扩展维度以支持广播 [H, W, 1, 2] -> [B, H, W, num_anchors, 2]
        grid_xy = grid_xy.view(1, H, W, 1, 2).expand(B, H, W, self.num_anchors, 2)

        # 调整锚框形状 [num_anchors, 2] -> [1, 1, 1, num_anchors, 2]
        anchor_wh = anchor_sizes.view(1, 1, 1, self.num_anchors, 2)

        # 将预测张量重塑为 [B, H, W, num_anchors, (5 + num_classes)]
        prediction = prediction.view(B, H, W, self.num_anchors, -1)

        # 解码坐标和尺寸 (向量化操作)
        prediction[..., :2] = (torch.sigmoid(prediction[..., :2]) + grid_xy) * stride
        prediction[..., 2:4] = anchor_wh * torch.exp(prediction[..., 2:4]) * stride
        prediction[..., 4:] = torch.sigmoid(prediction[..., 4:])

        # 重塑回原始维度 [B, H, W, C]
        prediction = prediction.view(B, H, W, -1)

        return prediction
