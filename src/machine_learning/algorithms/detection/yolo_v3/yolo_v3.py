from typing import Literal, Iterable
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
        - models (Mapping[str, BaseNet]): yolov3算法所需模型, {"darknet":model1, "fpn":model2}.
        - name (str): 算法名称. Default to "yolo_v3".
        - device (str): 运行设备(auto自动选择).
        """
        super().__init__(cfg=cfg, models=models, name=name, device=device)

        # 配置主要参数
        self.num_classes = self.cfg["algorithm"]["num_classes"]
        self.num_anchors = self.cfg["algorithm"]["num_anchors"]
        self.anchor_sizes = self.cfg["algorithm"]["anchor_sizes"]
        self.default_img_size = self.cfg["algorithm"]["image_size"]

        # loss权重参数
        self.b_weiget = self.cfg["algorithm"].get("b_weiget", 0.05)
        self.o_weiget = self.cfg["algorithm"].get("o_weiget", 1.0)
        self.c_weiget = self.cfg["algorithm"].get("c_weiget", 0.5)

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
        self.set_train()

        total_loss = 0.0

        for batch_idx, (img, bboxes, category_ids, indices) in enumerate(self.train_loader):
            img = img.to(self.device, non_blocking=True)
            cls_imgids_bboxes = torch.cat([indices, category_ids, bboxes], dim=-1).to(
                self.device
            )  # (class_id, img_id, bboxes)

            self._optimizers["yolo"].zero_grad()

            skips = self.models["darknet"](img)
            fimgs = self.models["fpn"](skips)

            # 特征图解码
            dets, anchor_sizes, strides = self.feature_decode(fimgs, img_size=img.shape[2])

            loss = self.criterion(dets, anchor_sizes, strides, bboxes, cls_imgids_bboxes)
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
        self.set_eval()

        total_loss = 0.0

        with torch.no_grad():
            for img, bboxes, category_ids, indices in self.train_loader:
                img = img.to(self.device, non_blocking=True)
                cls_imgids_bboxes = torch.cat([indices, category_ids, bboxes], dim=-1).to(
                    self.device
                )  # (class_id, img_id, bboxes)

                skips = self.models["darknet"](img)
                fimgs = self.models["fpn"](skips)

                # 特征图解码
                dets, anchor_sizes, strides = self.feature_decode(fimgs, img_size=img.shape[2])

                total_loss += self.criterion(dets, anchor_sizes, strides, cls_imgids_bboxes).item()

        avg_loss = total_loss / len(self.val_loader)

        return {"yolo": avg_loss, "save": avg_loss}

    def eval(self, num_samples: int = 5) -> None:
        self.set_eval()

    # 损失函数设计
    def criterion(
        self,
        dets: list[torch.Tensor],
        anchor_sizes: list[torch.Tensor],
        strides: list[int],
        cls_imgids_bboxes: torch.Tensor,
    ) -> torch.Tensor:
        # bbox_loss计算

        # obj_loss计算

        # cls_loss计算

        loss = self.b_weiget * bbox_loss + self.o_weiget * obj_loss + self.c_weiget * cls_loss

        return loss

    # 特征图解码
    def feature_decode(self, features: Iterable[torch.Tensor], img_size: int) -> tuple[list]:
        prediction_ls, anchor_sizes_ls, stride_ls = [], [], []

        for feature in features:
            # 调整维度顺序 [B,C,H,W] -> [B,H,W,C]
            prediction = feature.permute(0, 2, 3, 1).contiguous()
            B, H, W, _ = prediction.shape
            stride = img_size // H  # 计算步长

            # 根据特征图尺寸选择锚框
            if H == 52:
                anchor_sizes = torch.tensor(self.anchor_sizes[:3], dtype=torch.int32, device=self.device)
            elif H == 26:
                anchor_sizes = torch.tensor(self.anchor_sizes[3:6], dtype=torch.int32, device=self.device)
            else:
                anchor_sizes = torch.tensor(self.anchor_sizes[6:9], dtype=torch.int32, device=self.device)

            # 构建偏移矩阵
            grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
            grid_xy = torch.stack((grid_x, grid_y), dim=-1).to(prediction.device)  # [H, W, 2]

            # 扩展维度以支持广播 [H, W, 1, 2] -> [B, H, W, num_anchors, 2]
            grid_xy = grid_xy.view(1, H, W, 1, 2).expand(B, H, W, self.num_anchors, 2)

            # 调整锚框形状 [num_anchors, 2] -> [1, 1, 1, num_anchors, 2]
            anchor_wh = anchor_sizes.view(1, 1, 1, self.num_anchors, 2)

            # 将预测张量重塑为 [B, H, W, num_anchors, (5 + num_classes)]
            prediction = prediction.view(B, H, W, self.num_anchors, -1)

            # 特征解码
            prediction[..., :2] = (torch.sigmoid(prediction[..., :2]) + grid_xy) * stride
            prediction[..., 2:4] = anchor_wh * torch.exp(prediction[..., 2:4])
            prediction[..., 4] = torch.sigmoid(prediction[..., 4])
            prediction[..., 5:] = torch.sigmoid(prediction[..., 5:])

            # 重塑回原始维度 [B, H, W, C]
            prediction = prediction.view(B, H, W, -1)

            prediction_ls.append(prediction)
            anchor_sizes_ls.append(anchor_sizes)
            stride_ls.append(stride)

        return prediction_ls, anchor_sizes_ls, stride_ls

    def prepare_targets(
        self,
        dets: list[torch.Tensor],
        anchor_sizes: list[torch.Tensor],
        strides: list[int],
        cls_imgids_bboxes: torch.Tensor,
    ) -> tuple:
        """
        将原始图像目标信息映射到特征空间, 为loss计算做准备, 不将特征空间映射到原始图像空间的原因是
        特征空间的边界框数量远远大于目标边界框,比如52*52特征图的边界框数量为52*52*3, 计算复杂度高
        """
        num_bboxes = cls_imgids_bboxes.shape[0]  # number of bboxes(targets)

        for i, (det, anchor_size, strides) in enumerate(zip(dets, anchor_sizes, strides)):
            normalized_anchor_size = anchor_size / strides
            det_width, det_height = det.shape[2], det.shape[1]

            t = cls_imgids_bboxes.repeat(self.num_anchors, 1, 1)
            t[:, :, 2:6] *= cls_imgids_bboxes.new([det_width, det_height]).repeat(num_bboxes, 2)

            # 筛选合适的目标框
            if num_bboxes:
                r = t[:, :, 4:6] / normalized_anchor_size[:, None]
                j = torch.max(r, 1.0 / r).max(2)[0] < 4
                t = t[j]

            b, c = t[:, :2].long().T
            gxy = t[:, 2:4]
            gwh = t[:, 4:6]  # grid wh
            gij = gxy.long()
            gi, gj = gij.T  # grid xy indices

            # Convert anchor indexes to int
            a = t[:, 6].long()
            indices.append((b, a, gj.clamp_(0, gain[3].long() - 1), gi.clamp_(0, gain[2].long() - 1)))
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])
            tcls.append(c)

        return tcls, tbox, indices, anch
