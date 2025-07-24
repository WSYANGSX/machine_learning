from __future__ import annotations
from typing import Literal, Mapping, Any, Union

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from machine_learning.networks import BaseNet
from machine_learning.algorithms.base import AlgorithmBase
from machine_learning.types.aliases import FilePath
from machine_learning.modules.blocks import DFL
from machine_learning.utils.ops import empty_like
from ultralytics.utils.loss import TaskAlignedAssigner, BboxLoss


class YoloVMM(AlgorithmBase):
    def __init__(
        self,
        cfg: FilePath | Mapping[str, Any],
        models: Mapping[str, BaseNet],
        data: Mapping[str, Union[Dataset, Any]],
        name: str | None = "yolo_mm",
        device: Literal["cuda", "cpu", "auto"] = "auto",
    ) -> None:
        """
        Implementation of YoloMM object detection algorithm

        Args:
            cfg (str, dict): Configuration of the algorithm, it can be yaml file path or cfg dict.
            models (dict[str, BaseNet]): Models required by the YOLOMM algorithm, {"net": model}.
            data (Mapping[str, Union[Dataset, Any]]): Parsed specific dataset data, must including train dataset and val
            dataset, may contain data information of the specific dataset.
            name (str): Name of the algorithm. Defaults to "yolo_mm".
            device (Literal[&quot;cuda&quot;, &quot;cpu&quot;, &quot;auto&quot;], optional): Running device. Defaults to
            "auto"-automatic selection by algorithm.
        """
        super().__init__(cfg=cfg, models=models, data=data, name=name, device=device)

        # main parameters of the algorithm
        self.img_size = self.cfg["algorithm"].get("img_size", None)

        self.iou_threshold = self.cfg["algorithm"]["iou_threshold"]
        self.conf_threshold = self.cfg["algorithm"]["conf_threshold"]
        self.nms_threshold = self.cfg["algorithm"]["nms_threshold"]
        self.reg_max = self.cfg["algorithm"]["reg_max"]
        self.label_smoothing_scale = self.cfg["algorithm"]["label_smoothing_scale"]
        self.nc = self._models["nblitynet"].nc
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
        self.bbox_loss = BboxLoss(self.reg_max).to(device)
        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=device)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def _configure_optimizers(self) -> None:
        opt_cfg = self._cfg["optimizer"]

        g = [], [], []
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)

        for module_name, module in self.models["nblitynet"].named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn):  # weight (no decay)
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

        if opt_cfg["type"] == "Adam":
            optimizer = torch.optim.Adam(
                params=g[2],
                lr=opt_cfg["learning_rate"],
                betas=(opt_cfg["beta1"], opt_cfg["beta2"]),
                eps=opt_cfg["eps"],
                weight_decay=0.0,
            )
        elif opt_cfg["type"] == "SGD":
            optimizer = torch.optim.SGD(
                params=g[2], lr=opt_cfg["learning_rate"], momentum=opt_cfg["momentum"], weight_decay=0.0, nesterov=True
            )

        else:
            raise ValueError(f"Does not support optimizer:{opt_cfg['type']} currently.")

        optimizer.add_param_group({"params": g[0], "weight_decay": self._cfg["optimizer"].get("weight_decay", 1e-5)})
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})

        self._optimizers["yolo_mm"] = optimizer

    def _configure_schedulers(self) -> None:
        sch_config = self._cfg["scheduler"]

        if sch_config.get("type") == "ReduceLROnPlateau":
            self._schedulers.update(
                {
                    "yolo_mm": torch.optim.lr_scheduler.ReduceLROnPlateau(
                        self._optimizers["yolo_mm"],
                        mode="min",
                        factor=sch_config.get("factor", 0.1),
                        patience=sch_config.get("patience", 10),
                    )
                }
            )

        if sch_config.get("type") == "CustomLRDecay":
            self._schedulers.update(
                {
                    "yolo_mm": torch.optim.lr_scheduler.LambdaLR(
                        self._optimizers["yolo_mm"],
                        lr_lambda=lambda x: max(1 - x / sch_config["final_epoch"], 0)
                        * (1.0 - sch_config["learning_rate_final_factor"])
                        + sch_config["learning_rate_final_factor"],
                    )
                }
            )

        else:
            print(f"Warning: Unknown scheduler type '{sch_config.get('type')}', no scheduler configured.")

    def train_epoch(self, epoch: int, writer: SummaryWriter, log_interval: int = 10) -> dict[str, float]:
        self.set_train()

        total_loss = 0.0
        scaler = GradScaler()

        for batch_idx, (imgs, thermals, iid_cls_bboxes) in enumerate(self.train_loader):
            imgs = imgs.to(self.device, non_blocking=True)
            thermals = thermals.to(self.device, non_blocking=True)
            targets = iid_cls_bboxes.to(self.device)  # (img_ids, class_ids, bboxes)

            self._optimizers["yolo_mm"].zero_grad()

            with autocast(
                device_type=str(self.device)
            ):  # Ensure that the autocast scope correctly covers the forward computation
                pred1, pred2, pred3 = self.models["nblitynet"](imgs, thermals)
                loss, loss_components = self.criterion(
                    preds=[pred1, pred2, pred3], targets=targets, img_size=imgs.size(2)
                )

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.params, self._cfg["optimizer"]["grad_clip"])
            scaler.step(self._optimizers["yolo_mm"])
            scaler.update()

            total_loss += loss.item()

            batches = epoch * len(self.train_loader) + batch_idx
            if batch_idx % log_interval == 0:
                writer.add_scalar("box_loss/train_batch", loss_components["box_loss"], batches)
                writer.add_scalar("cls_loss/train_batch", loss_components["cls_loss"], batches)
                writer.add_scalar("dfl_loss/train_batch", loss_components["dfl_loss"], batches)
                writer.add_scalar("total_loss/train_batch", loss_components["total_loss"], batches)

        avg_loss = total_loss / len(self.train_loader)

        return {"yolo_mm loss": avg_loss}

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        self.set_eval()

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
