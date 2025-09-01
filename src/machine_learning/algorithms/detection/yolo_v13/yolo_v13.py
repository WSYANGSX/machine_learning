from __future__ import annotations
from typing import Literal, Mapping, Any, Sequence, Union
from tqdm import tqdm

import torch
import numpy as np
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from machine_learning.networks import BaseNet
from machine_learning.types.aliases import FilePath
from machine_learning.utils.logger import LOGGER
from machine_learning.utils.layers import NORM_LAYER_TYPES
from machine_learning.algorithms.base import AlgorithmBase
from ultralytics.utils.loss import TaskAlignedAssigner, BboxLoss
from machine_learning.utils.detection import non_max_suppression, box_iou, xywh2xyxy, compute_ap, match_predictions


class YoloV13(AlgorithmBase):
    def __init__(
        self,
        cfg: FilePath | Mapping[str, Any],
        data: Mapping[str, Union[Dataset, Any]],
        net: BaseNet,
        name: str | None = "yolov13",
        device: Literal["cuda", "cpu", "auto"] = "auto",
        amp: bool = True,
    ) -> None:
        """
        Implementation of YoloV13 object detection algorithm

        Args:
            cfg (FilePath, Mapping[str, Any]): Configuration of the algorithm, it can be yaml file path or cfg dict.
            data (Mapping[str, Union[Dataset, Any]]): Parsed specific dataset data, must including train dataset and val
            dataset, may contain data information of the specific dataset.
            net (BaseNet): Models required by the YoloV13 algorithm.
            name (str): Name of the algorithm. Defaults to "yolov13".
            device (Literal[&quot;cuda&quot;, &quot;cpu&quot;, &quot;auto&quot;], optional): Running device. Defaults to
            "auto"-automatic selection by algorithm.
        """
        super().__init__(cfg=cfg, net=net, name=name, device=device, data=data, amp=amp)

        # main parameters of the algorithm
        self.img_size = self.cfg["algorithm"].get("img_size", None)
        self.iou_thres = self.cfg["algorithm"]["iou_thres"]
        self.conf_thres_val = self.cfg["algorithm"]["conf_thres_val"]
        self.conf_thres_det = self.cfg["algorithm"]["conf_thres_det"]
        self.reg_max = self.cfg["algorithm"]["reg_max"]
        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=self.device)
        self.nc = self.net.nc
        self.no = self.nc + self.reg_max * 4
        self.close_mosaic_epoch = self.cfg["algorithm"]["close_mosaic_epoch"]

        self.topk = self.cfg["algorithm"]["topk"]
        self.alpha = self.cfg["algorithm"]["alpha"]
        self.beta = self.cfg["algorithm"]["beta"]
        self.box_weight = self.cfg["algorithm"].get("box")
        self.cls_weight = self.cfg["algorithm"].get("cls")
        self.dfl_weight = self.cfg["algorithm"].get("dfl")

        self.use_dfl = self.reg_max > 1
        self.assigner = TaskAlignedAssigner(topk=self.topk, num_classes=self.nc, alpha=self.alpha, beta=self.beta)
        self.bbox_loss = BboxLoss(self.reg_max).to(self.device)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()

    def _configure_optimizers(self) -> None:
        self.opt_cfg = self._cfg["optimizer"]

        self.optimizer = None

        decay_params = []
        no_decay_norm_params = []
        no_decay_bias_params = []

        for name, param in self.net.named_parameters():
            # Skip the freezed parameter
            if not param.requires_grad:
                continue

            full_name = name
            module_name = name.rsplit(".", 1)[0] if "." in name else ""
            module = dict(self.net.named_modules()).get(module_name, None)

            if "bias" in full_name:
                no_decay_bias_params.append(param)
            elif module is not None and isinstance(module, NORM_LAYER_TYPES):
                no_decay_norm_params.append(param)
            else:
                decay_params.append(param)

        # Set the weight attenuation value
        weight_decay = self._cfg["optimizer"].get("weight_decay", 1e-5)

        # Create parameter groups
        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_norm_params, "weight_decay": 0.0},
            {"params": no_decay_bias_params, "weight_decay": 0.0},
        ]

        optimizer_type = self.opt_cfg["type"]
        lr = self.opt_cfg["learning_rate"]

        if optimizer_type == "Adam":
            self.optimizer = torch.optim.Adam(
                param_groups,
                lr=lr,
                betas=(self.opt_cfg["beta1"], self.opt_cfg["beta2"]),
                eps=self.opt_cfg["eps"],
            )
        elif optimizer_type == "SGD":
            momentum = self.opt_cfg["momentum"]
            nesterov = momentum > 0

            self.optimizer = torch.optim.SGD(
                param_groups,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,  # overwritten by the parameter group
                nesterov=nesterov,
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

        self._add_optimizer("optimizer", self.optimizer)

    def _configure_schedulers(self) -> None:
        self.sch_config = self._cfg["scheduler"]

        self.scheduler = None

        if self.sch_config.get("type") == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.sch_config.get("factor", 0.1),
                patience=self.sch_config.get("patience", 10),
            )
            self._add_scheduler("scheduler", self.scheduler)

        elif self.sch_config.get("type") == "CustomLRDecay":
            self.lf = (
                lambda x: max(1 - x / self.cfg["train"]["epochs"], 0) * (1.0 - self.opt_cfg["final_factor"])
                + self.opt_cfg["final_factor"]
            )  # linear
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
            self._add_scheduler("scheduler", self.scheduler)

        else:
            LOGGER.warning(f"Unknown scheduler type '{self.sch_config.get('type')}', no scheduler configured.")

    def train_epoch(self, epoch: int, writer: SummaryWriter, log_interval: int = 10) -> dict[str, float]:
        self.set_train()

        # close mosaic
        self.close_mosaic(epoch)

        total_loss = 0.0

        pbar = tqdm(enumerate(self.train_loader), total=self.num_batches)
        for batch_idx, (imgs, iid_cls_bboxes) in pbar:
            # Warmup
            batch_inters = epoch * self.num_batches + batch_idx
            self.warmup(batch_inters, epoch)

            # Load data
            imgs = imgs.to(self.device, non_blocking=True)
            targets = iid_cls_bboxes.to(self.device)  # (img_ids, class_ids, bboxes)

            # Loss calculation
            with autocast(enabled=self.amp):  # Ensure that the autocast scope correctly covers the forward computation
                pred1, pred2, pred3 = self.net(imgs)
                loss, loss_components = self.criterion(
                    preds=[pred1, pred2, pred3], targets=targets, img_size=imgs.size(2)
                )

            # Gradient backpropagation
            self.backward(loss)
            # Parameter optimization
            self.optimizer_step(batch_inters)

            total_loss += loss.item()

            if batch_idx % log_interval == 0:
                writer.add_scalar("box_loss/train_batch", loss_components["box_loss"], batch_inters)
                writer.add_scalar("cls_loss/train_batch", loss_components["cls_loss"], batch_inters)
                writer.add_scalar("dfl_loss/train_batch", loss_components["dfl_loss"], batch_inters)

        avg_loss = total_loss / len(self.train_loader)

        return {"loss": avg_loss}

    def close_mosaic(self, epoch: int) -> None:
        if epoch == self.close_mosaic_epoch:
            if hasattr(self.train_loader.dataset, "mosaic"):
                LOGGER.info("Closing dataloader mosaic...")
                self.train_loader.dataset.mosaic = False

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        self.set_eval()

        metrics = {}
        total_loss, total_box_loss, total_cls_loss, total_dfl_loss = 0.0, 0.0, 0.0, 0.0

        # Initiate the evaluation indicators
        stats = []
        for _, (imgs, targets) in enumerate(self.val_loader):
            imgs = imgs.to(self.device, non_blocking=True)
            targets = targets.to(self.device)  # (img_ids, class_ids, bboxes)
            img_size = imgs.size(2)

            preds = self.net(imgs)
            loss, loss_components = self.criterion(preds=preds, targets=targets, img_size=img_size)

            total_loss += loss.item()
            total_box_loss += loss_components["box_loss"]
            total_cls_loss += loss_components["cls_loss"]
            total_dfl_loss += loss_components["dfl_loss"]

            # Merge multi-scale predictions
            x = []
            for pred in preds:
                bs, _, h, w = pred.shape
                pred = pred.permute(0, 2, 3, 1).reshape(bs, h * w, -1)
                x.append(pred)
            x = torch.cat(x, 1)
            # Separate the bounding box from the category score
            box, cls = x.split((self.reg_max * 4, self.nc), 2)

            # decode preds
            strides = torch.tensor([img_size // pred.size(2) for pred in preds], device=self.device)
            anchor_points, stride_tensor = make_anchors(preds, strides, 0.5)
            dbox = self.bbox_decode(anchor_points, box, xywh=True) * stride_tensor
            prediction = torch.cat((dbox, cls.sigmoid()), 2)

            # NMS
            preds = non_max_suppression(
                prediction.permute(0, 2, 1), conf_thres=self.conf_thres_val, iou_thres=self.iou_thres
            )

            # prepare targets
            scale = torch.tensor([imgs.shape[3], imgs.shape[2]] * 2, device=self.device)
            targets_abs = targets.clone()
            bbox_abs = targets[:, 2:6] * scale  # convert to abs xywh
            targets_abs[:, 2:6] = xywh2xyxy(bbox_abs)  # convert to xyxy

            for si, pred in enumerate(preds):
                # get the real frame of the current image (img_id, class_id, x1, y1, x2, y2)
                labels = targets_abs[targets_abs[:, 0] == si, 1:]

                # extract the category and coordinates of the real box
                tcls = labels[:, 0] if len(labels) > 0 else torch.empty(0, device=self.device)
                tbox = labels[:, 1:5] if len(labels) > 0 else torch.empty(0, 4, device=self.device)

                if len(pred) == 0:
                    if len(labels):
                        stats.append(
                            (
                                torch.zeros(0, self.niou, dtype=torch.bool),  # TP
                                torch.Tensor(),  # confidence scores
                                torch.Tensor(),  # pred_classes
                                tcls.cpu(),  # ground truth classes
                            )
                        )
                    continue

                # Prediction box format: [x1, y1, x2, y2, conf, class]
                pred_boxes = pred[:, :4]
                pred_scores = pred[:, 4]
                pred_classes = pred[:, 5]

                # calculate IoU
                iou = box_iou(tbox, pred_boxes)  # shape: [n_pred, n_gt]
                # Find the best matching real box for each prediction box
                tp = match_predictions(pred_classes, tcls, iou, self.iouv)

                # record statistical information
                stats.append((tp.cpu(), pred_scores.cpu(), pred_classes.cpu(), tcls.cpu()))

        # calculate the average loss
        num_batches = len(self.val_loader)
        metrics["loss"] = total_loss / num_batches
        metrics["box_loss"] = total_box_loss / num_batches
        metrics["cls_loss"] = total_cls_loss / num_batches
        metrics["dfl_loss"] = total_dfl_loss / num_batches
        metrics["save metric"] = metrics["loss"]

        # calculate mAP metrics
        if stats:
            tp, conf, pred_cls, target_cls = zip(*stats)
            tp = np.concatenate(tp, 0)
            conf = np.concatenate(conf, 0)
            pred_cls = np.concatenate(pred_cls, 0)
            target_cls = np.concatenate(target_cls, 0)
            metrics.update(compute_ap(self.cfg["data"]["class_names"], tp, conf, pred_cls, target_cls, self.iouv))

        return metrics

    def warmup(self, batch_inters: int, epoch: int) -> int:
        nw = (
            max(round(self.opt_cfg["warmup_epochs"] * self.num_batches), 100)
            if self.opt_cfg["warmup_epochs"] > 0
            else -1
        )  # warmup batches

        if batch_inters <= nw:
            xi = [0, nw]  # x interp
            self.accumulate = max(
                1, int(np.interp(batch_inters, xi, [1, self.nominal_batch_size / self.batch_size]).round())
            )
            for j, x in enumerate(self.optimizer.param_groups):
                # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                x["lr"] = np.interp(
                    batch_inters,
                    xi,
                    [self.opt_cfg["warmup_bias_lr"] if j == 2 else 0.0, x["initial_lr"] * self.lf(epoch)],
                )
                if "momentum" in x:
                    x["momentum"] = np.interp(
                        batch_inters, xi, [self.opt_cfg["warmup_momentum"], self.opt_cfg["momentum"]]
                    )

    def eval(self) -> None:
        raise NotImplementedError(
            "Eval method is not implemented in the yolo-series algorithms, "
            + "please use detect method to detect objects in an image."
        )

    def criterion(self, preds: Sequence[torch.Tensor], targets: torch.Tensor, img_size: int):
        cls_loss = torch.zeros(1, device=self.device)
        box_loss = torch.zeros(1, device=self.device)
        dfl_loss = torch.zeros(1, device=self.device)

        strides = torch.tensor([img_size // pred.size(2) for pred in preds], device=self.device)
        pred_distri, pred_scores = torch.cat([xi.view(preds[0].shape[0], self.no, -1) for xi in preds], 2).split(
            (self.reg_max * 4, self.nc), 1
        )  # [bs, no, h1*w1+h2*w2+h3*w3] -> [bs, 4*reg_max, h1*w1+h2*w2+h3*w3] & [bs, nc, h1*w1+h2*w2+h3*w3]

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()  # [bs, h1*w1+h2*w2+h3*w3, nc]
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()  # [bs, h1*w1+h2*w2+h3*w3, 4*reg_max]

        bs = pred_scores.shape[0]
        anchor_points, stride_tensor = make_anchors(preds, strides, 0.5)

        # Targets
        scale_tensor = torch.tensor([img_size] * 4, device=self.device)
        targets = torch.cat((targets[:, [0]], targets[:, [1]], targets[:, 2:6]), 1)
        targets = self.preprocess(targets, bs, scale_tensor=scale_tensor)
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

        total_loss = (box_loss + cls_loss + dfl_loss) * bs

        loss_component = {
            "box_loss": box_loss.item(),
            "cls_loss": cls_loss.item(),
            "dfl_loss": dfl_loss.item(),
        }

        return total_loss, loss_component

    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor, xywh: bool = False):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=xywh)

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

    def detect(self, img: torch.Tensor, thermal: torch.Tensor) -> None:
        pass


"""
Helper functions
"""


def make_anchors(preds, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert preds is not None
    dtype, device = preds[0].dtype, preds[0].device
    for i, stride in enumerate(strides):
        h, w = preds[i].shape[2:] if isinstance(preds, (list, tuple)) else (int(preds[i][0]), int(preds[i][1]))
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
