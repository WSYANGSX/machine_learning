from __future__ import annotations
from typing import Literal, Mapping, Any

import torch
import torchvision
import numpy as np
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from machine_learning.networks import BaseNet
from machine_learning.algorithms.base import AlgorithmBase
from machine_learning.types.aliases import FilePath
from machine_learning.modules.blocks import DFL
from machine_learning.utils.ops import empty_like
from ultralytics.utils.loss import TaskAlignedAssigner, BboxLoss


class YoloMM(AlgorithmBase):
    def __init__(
        self,
        cfg: FilePath | Mapping[str, Any],
        net: BaseNet,
        name: str | None = "yolo_mm",
        device: Literal["cuda", "cpu", "auto"] = "auto",
    ) -> None:
        """
        Implementation of YoloMM object detection algorithm

        Args:
            cfg (FilePath, Mapping[str, Any]): Configuration of the algorithm, it can be yaml file path or cfg dict.
            net (BaseNet): Models required by the YOLOMM algorithm.
            name (str): Name of the algorithm. Defaults to "yolo_mm".
            device (Literal[&quot;cuda&quot;, &quot;cpu&quot;, &quot;auto&quot;], optional): Running device. Defaults to
            "auto"-automatic selection by algorithm.
        """
        super().__init__(cfg=cfg, net=net, name=name, device=device)

        # main parameters of the algorithm
        self.img_size = self.cfg["algorithm"].get("img_size", None)

        self.iou_threshold = self.cfg["algorithm"]["iou_threshold"]
        self.conf_threshold = self.cfg["algorithm"]["conf_threshold"]
        self.nms_threshold = self.cfg["algorithm"]["nms_threshold"]
        self.reg_max = self.cfg["algorithm"]["reg_max"]
        self.label_smoothing_scale = self.cfg["algorithm"]["label_smoothing_scale"]
        self.nc = self.net.nc
        self.no = self.nc + self.reg_max * 4

        self.topk = self.cfg["algorithm"]["topk"]
        self.alpha = self.cfg["algorithm"]["alpha"]
        self.beta = self.cfg["algorithm"]["beta"]

        self.box_weight = self.cfg["algorithm"].get("box", 0.05)
        self.cls_weight = self.cfg["algorithm"].get("cls", 1.0)
        self.dfl_weight = self.cfg["algorithm"].get("dfl", 0.5)

        self.use_dfl = self.reg_max > 1
        self.dfl = DFL(self.reg_max) if self.use_dfl else nn.Identity()
        self.assigner = TaskAlignedAssigner(topk=self.topk, num_classes=self.nc, alpha=self.alpha, beta=self.beta)
        self.bbox_loss = BboxLoss(self.reg_max).to(self.device)
        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=self.device)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def _configure_optimizers(self) -> None:
        self.opt_cfg = self._cfg["optimizer"]

        g = [], [], []
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)

        for module_name, module in self.net.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn):  # weight (no decay)
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

        if self.opt_cfg["type"] == "Adam":
            self.optimizer = torch.optim.Adam(
                params=g[2],
                lr=self.opt_cfg["learning_rate"],
                betas=(self.opt_cfg["beta1"], self.opt_cfg["beta2"]),
                eps=self.opt_cfg["eps"],
                weight_decay=0.0,
            )
        elif self.opt_cfg["type"] == "SGD":
            self.optimizer = torch.optim.SGD(
                params=g[2],
                lr=self.opt_cfg["learning_rate"],
                momentum=self.opt_cfg["momentum"],
                weight_decay=0.0,
                nesterov=True,
            )

        else:
            raise ValueError(f"Does not support optimizer:{self.opt_cfg['type']} currently.")

        self.optimizer.add_param_group(
            {"params": g[0], "weight_decay": self._cfg["optimizer"].get("weight_decay", 1e-5)}
        )
        self.optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})

        self._add_optimizer("optimizer", self.optimizer)

    def _configure_schedulers(self) -> None:
        self.sch_config = self._cfg["scheduler"]

        if self.sch_config.get("type") == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.sch_config.get("factor", 0.1),
                patience=self.sch_config.get("patience", 10),
            )

        if self.sch_config.get("type") == "CustomLRDecay":
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda x: max(1 - x / self.cfg["trainer"]["epochs"], 0)
                * (1.0 - self.opt_cfg["learning_rate_final_factor"])
                + self.opt_cfg["learning_rate_final_factor"],
            )

        else:
            print(f"Warning: Unknown scheduler type '{self.sch_config.get('type')}', no scheduler configured.")

    def train_epoch(self, epoch: int, writer: SummaryWriter, log_interval: int = 10) -> dict[str, float]:
        self.set_train()

        total_loss = 0.0
        scaler = GradScaler()

        for batch_idx, (imgs, thermals, iid_cls_bboxes) in enumerate(self.train_loader):
            imgs = imgs.to(self.device, non_blocking=True)
            thermals = thermals.to(self.device, non_blocking=True)
            targets = iid_cls_bboxes.to(self.device)  # (img_ids, class_ids, bboxes)

            self.optimizer.zero_grad()

            with autocast(
                enabled=True, dtype=torch.float16
            ):  # Ensure that the autocast scope correctly covers the forward computation
                pred1, pred2, pred3 = self.net(imgs, thermals)
                loss, loss_components = self.criterion(
                    preds=[pred1, pred2, pred3], targets=targets, img_size=imgs.size(2)
                )

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self._cfg["optimizer"]["grad_clip"])
            scaler.step(self.optimizer)
            scaler.update()

            total_loss += loss.item()

            batches = epoch * len(self.train_loader) + batch_idx
            if batch_idx % log_interval == 0:
                writer.add_scalar("box_loss/train_batch", loss_components["box_loss"], batches)
                writer.add_scalar("cls_loss/train_batch", loss_components["cls_loss"], batches)
                writer.add_scalar("dfl_loss/train_batch", loss_components["dfl_loss"], batches)
                writer.add_scalar("total_loss/train_batch", loss_components["total_loss"], batches)

        avg_loss = total_loss / len(self.train_loader)

        return {"loss": avg_loss}

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        self.set_eval()

        metrics = {}

        total_loss = 0.0
        total_box_loss = 0.0
        total_cls_loss = 0.0
        total_dfl_loss = 0.0

        # 初始化评估指标
        stats = []
        iouv = torch.linspace(0.5, 0.95, 10, device=self.device)  # iou从0.5到0.95，步长0.05
        niou = iouv.numel()

        for batch_idx, (imgs, thermals, targets) in enumerate(self.val_loader):
            imgs = imgs.to(self.device, non_blocking=True)
            thermals = thermals.to(self.device, non_blocking=True)
            targets = targets.to(self.device)  # (img_ids, class_ids, bboxes)

            # 前向传播
            with autocast(enabled=True, dtype=torch.float16):
                pred1, pred2, pred3 = self.net(imgs, thermals)
                loss, loss_components = self.criterion(
                    preds=[pred1, pred2, pred3], targets=targets, img_size=imgs.size(2)
                )

                # 计算预测结果
                preds = self.non_max_suppression([pred1, pred2, pred3])

        total_loss += loss.item()
        total_box_loss += loss_components["box_loss"]
        total_cls_loss += loss_components["cls_loss"]
        total_dfl_loss += loss_components["dfl_loss"]

        targets[:, 2:] *= torch.tensor((imgs.shape[3], imgs.shape[2], imgs.shape[3], imgs.shape[2]), device=self.device)

        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]  # 当前图片的标签
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # 类别列表

            # 如果没有预测结果，记录所有标签为未检测到
            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # 计算预测框与真实框的IoU
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=self.device)
            if nl:
                detected = []  # 已检测到的目标索引
                tcls_tensor = labels[:, 0]

                # 计算每个预测框与所有真实框的IoU
                ious, i = self.box_iou(pred[:, :4], labels[:, 1:]).max(1)

                # 选择IoU大于阈值的预测框
                j = (ious > self.iou_threshold) & (pred[:, 5] == tcls_tensor[i])

                # 每个真实目标只匹配一个预测框
                for k in j.nonzero(as_tuple=False):
                    if k not in detected:
                        detected.append(k)
                        correct[k] = ious[k] > iouv

            # 记录统计信息
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # 计算平均损失
        num_batches = len(self.val_loader)
        metrics["val/loss"] = total_loss / num_batches
        metrics["val/box_loss"] = total_box_loss / num_batches
        metrics["val/cls_loss"] = total_cls_loss / num_batches
        metrics["val/dfl_loss"] = total_dfl_loss / num_batches

        # 计算mAP指标
        if stats:
            tp, conf, pred_cls, target_cls = zip(*stats)
            metrics.update(self.compute_ap(tp, conf, pred_cls, target_cls))

        return metrics

    @torch.no_grad()
    def non_max_suppression(self, preds: list[torch.Tensor]) -> list[torch.Tensor]:
        """应用非极大值抑制处理预测结果"""
        output = [torch.empty((0, 6), device=self.device)] * preds[0].shape[0]

        # 合并多尺度预测
        x = []
        for pred in preds:
            bs, _, ny, nx = pred.shape
            pred = pred.permute(0, 2, 3, 1).reshape(bs, ny * nx, -1)
            x.append(pred)
        x = torch.cat(x, 1)

        # 分割边界框和类别分数
        box, cls = x.split((self.reg_max * 4, self.nc), 2)
        dbox = dist2bbox(self.dfl(box), self.anchors, xywh=False, dim=1) * self.strides

        # 计算类别分数
        scores = cls.sigmoid()

        # 处理每张图片
        for i in range(x.shape[0]):
            boxes = dbox[i]
            scores_i = scores[i]

            # 过滤低置信度预测
            conf_mask = scores_i.max(1)[0] > self.conf_threshold
            boxes, scores_i = boxes[conf_mask], scores_i[conf_mask]

            if not boxes.shape[0]:
                continue

            # 非极大值抑制
            boxes = boxes.clone()
            scores_i, classes = scores_i.max(1)
            boxes = torch.cat((boxes, scores_i.unsqueeze(1), classes.unsqueeze(1)), 1)
            nms_mask = torchvision.ops.nms(boxes[:, :4], boxes[:, 4], self.nms_threshold)
            output[i] = boxes[nms_mask]

        return output

    def compute_ap(self, tp, conf, pred_cls, target_cls):
        """计算平均精度指标(mAP)"""
        # 按置信度排序所有预测结果
        i = np.argsort(-np.concatenate(conf))
        tp, conf, pred_cls = [np.concatenate(x, 0)[i] for x in [tp, conf, pred_cls]]

        # 计算PR曲线和AP值
        nc = self.nc  # 类别数量
        ap = np.zeros((nc, 10))
        p = np.zeros((nc, 1000))
        r = np.zeros((nc, 1000))

        # 为每个类别计算AP
        for ci in range(nc):
            # 获取当前类别的预测和标签
            ci_tp = tp[:, ci]
            ci_pred_cls = pred_cls == ci
            if ci_pred_cls.sum() == 0:
                continue

            # 计算精度和召回率
            n_gt = (np.concatenate(target_cls, 0) == ci).sum()
            n_p = ci_pred_cls.sum()

            if n_gt == 0 or n_p == 0:
                continue

            # 累积FP和TP
            fpc = (1 - ci_tp[ci_pred_cls]).cumsum()
            tpc = ci_tp[ci_pred_cls].cumsum()

            # 召回率
            recall = tpc / (n_gt + 1e-16)
            r[ci] = np.interp(-np.linspace(0, 1, 1000), -conf[ci_pred_cls], recall)

            # 精度
            precision = tpc / (tpc + fpc)
            p[ci] = np.interp(-np.linspace(0, 1, 1000), -conf[ci_pred_cls], precision)

            # AP值 (积分PR曲线)
            for j in range(10):
                ap[ci, j] = compute_ap_single(precision, recall, iouv=j * 0.05 + 0.5)

        # 计算全局指标
        results = {
            "val/mAP": ap[:, :].mean(),
            "val/mAP_50": ap[:, 0].mean(),
            "val/mAP_75": ap[:, 5].mean(),
            "val/precision": p.mean(0).mean(),
            "val/recall": r.mean(0).mean(),
        }

        # 添加每个类别的AP
        for ci in range(nc):
            results[f"val/AP_{self.class_names[ci]}"] = ap[ci].mean()

        return results

    def box_iou(self, box1, box2):
        """计算两组边界框之间的IoU"""
        # 获取边界框坐标
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

        # 计算交集区域
        inter_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1)
        inter_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1)
        inter_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2)
        inter_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2)

        # 交集面积
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

        # 并集面积
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union_area = b1_area.unsqueeze(1) + b2_area - inter_area

        # 返回IoU
        return inter_area / (union_area + 1e-7)

    def eval(self) -> None:
        raise NotImplementedError(
            "Eval method is not implemented in the yolo-series algorithms, "
            + "please use detect method to detect objects in an image."
        )

    def criterion(self, preds: list[torch.Tensor], targets: torch.Tensor, img_size: int):
        cls_loss = torch.zeros(1, device=self.device)
        box_loss = torch.zeros(1, device=self.device)
        dfl_loss = torch.zeros(1, device=self.device)

        strides = torch.tensor([img_size // pred.size(2) for pred in preds], device=self.device)
        pred_distri, pred_scores = torch.cat([xi.view(preds[0].shape[0], self.no, -1) for xi in preds], 2).split(
            (self.reg_max * 4, self.nc), 1
        )  # [bs, no, h1*w1+h2*w2+h3*w3]

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()  # [bs, h1*w1+h2*w2+h3*w3, nc]
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()  # [bs, h1*w1+h2*w2+h3*w3, 4*reg_max]

        bs = pred_scores.shape[0]
        anchor_points, stride_tensor = make_anchors(preds, strides, 0.5)

        # Targets
        scale_tensor = torch.tensor([img_size] * 4, device=self.device)
        targets = torch.cat((targets[:, [0]], targets[:, [1]], targets[:, 2:6]), 1)
        targets = self.preprocess(targets.to(self.device), bs, scale_tensor=scale_tensor)
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        cls_loss = self.bce(pred_scores, target_scores.to(pred_scores.dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            box_loss, dfl_loss = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        cls_loss *= self.cls_weight  # box gain
        box_loss *= self.box_weight  # cls gain
        dfl_loss *= self.dfl_weight  # dfl gain

        total_loss = box_loss + cls_loss + dfl_loss

        loss_component = {
            "box_loss": box_loss.item(),
            "cls_loss": cls_loss.item(),
            "dfl_loss": dfl_loss.item(),
            "total_loss": total_loss.item(),
        }

        return total_loss, loss_component

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))

        return out


"""
Helper functions
"""


def compute_ap_single(precision, recall, iou_threshold=0.5):
    """计算单个IoU阈值下的平均精度"""
    # 平滑PR曲线
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # 找到召回率变化的点
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # 积分计算AP
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner. Note: ops per 2 channels faster than per channel.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)  # faster than clone/copy
    xy = x[..., :2]  # centers
    wh = x[..., 2:] / 2  # half width-height
    y[..., :2] = xy - wh  # top left xy
    y[..., 2:] = xy + wh  # bottom right xy
    return y


def make_anchors(preds, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert preds is not None
    dtype, device = preds[0].dtype, preds[0].device
    for i, stride in enumerate(strides):
        h, w = preds[i].shape[2:] if isinstance(preds, list) else (int(preds[i][0]), int(preds[i][1]))
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)  # dist (lt, rb)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox
