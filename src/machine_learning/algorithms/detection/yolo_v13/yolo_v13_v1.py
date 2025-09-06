from typing import Literal, Mapping, Any, Sequence, Union
from tqdm import tqdm

import torch
import numpy as np
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from machine_learning.networks import BaseNet
from machine_learning.types.aliases import FilePath
from machine_learning.utils.logger import LOGGER
from machine_learning.utils.layers import NORM_LAYER_TYPES
from machine_learning.algorithms.base import AlgorithmBase
from ultralytics.utils.loss import TaskAlignedAssigner, BboxLoss
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics
from machine_learning.utils.detection import non_max_suppression, box_iou, xywh2xyxy, match_predictions


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
            amp (bool): Whether to enable Automatic Mixed Precision. Defaults to False.
        """
        super().__init__(cfg=cfg, net=net, name=name, device=device, data=data, amp=amp)

        # main parameters of the algorithm
        self.iou_thres = self.cfg["algorithm"]["iou_thres"]
        self.conf_thres_val = self.cfg["algorithm"]["conf_thres_val"]
        self.conf_thres_det = self.cfg["algorithm"]["conf_thres_det"]
        self.reg_max = self.cfg["algorithm"]["reg_max"]
        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=self.device)
        self.nc = self.net.nc
        self.no = self.nc + self.reg_max * 4
        self.close_mosaic_epoch = self.cfg["algorithm"]["close_mosaic_epoch"]
        self.max_det = self.cfg["algorithm"]["max_det"]
        self.single_cls = self.cfg["algorithm"]["single_cls"]
        self.plots = self.cfg["algorithm"]["plots"]

        # weight
        self.box_weight = self.cfg["algorithm"].get("box")
        self.cls_weight = self.cfg["algorithm"].get("cls")
        self.dfl_weight = self.cfg["algorithm"].get("dfl")

        self.use_dfl = self.reg_max > 1
        self.topk = self.cfg["algorithm"]["topk"]
        self.alpha = self.cfg["algorithm"]["alpha"]
        self.beta = self.cfg["algorithm"]["beta"]
        self.assigner = TaskAlignedAssigner(topk=self.topk, num_classes=self.nc, alpha=self.alpha, beta=self.beta)
        self.bbox_loss = BboxLoss(self.reg_max).to(self.device)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

        # IoU vector for mAP@0.5:0.95
        self.iouv = torch.linspace(0.5, 0.95, 10)
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
        super().train_epoch(epoch, writer, log_interval)

        # close mosaic
        self.close_mosaic(epoch)

        # metrics
        metrics = {
            "tloss": None,
            "bloss": None,
            "dloss": None,
            "closs": None,
            "instances": None,
            "img_size": None,
        }
        self.print_metric_titles("train", metrics)

        tloss = None

        pbar = tqdm(enumerate(self.train_loader), total=self.train_batches)
        for batch_idx, (imgs, targets) in pbar:
            # Warmup
            batch_inters = epoch * self.train_batches + batch_idx
            self.warmup(batch_inters, epoch)

            # Load data
            imgs = imgs.to(self.device, non_blocking=True)
            targets = targets.to(self.device)  # (img_ids, class_ids, bboxes)

            # Loss calculation
            with autocast(
                device_type=str(self.device), enabled=self.amp
            ):  # Ensure that the autocast scope correctly covers the forward computation
                preds = self.net(imgs)
                loss, lc = self.criterion(preds=preds, targets=targets, imgs_shape=imgs.shape)

            # Gradient backpropagation
            self.backward(loss)
            # Parameter optimization
            self.optimizer_step(batch_inters)

            # Losses
            tloss = (tloss * batch_idx + loss.item()) / (batch_idx + 1) if tloss is not None else loss.item()
            bloss = lc["bloss"]
            closs = lc["closs"]
            dloss = lc["dloss"]

            # Metrics
            metrics["tloss"] = tloss
            metrics["bloss"] = bloss
            metrics["closs"] = closs
            metrics["dloss"] = dloss
            metrics["img_size"] = imgs.size(2)
            metrics["instances"] = targets.size(0)

            if batch_idx % log_interval == 0:
                writer.add_scalar("bloss/train_batch", bloss, batch_inters)
                writer.add_scalar("closs/train_batch", closs, batch_inters)
                writer.add_scalar("dloss/train_batch", dloss, batch_inters)

            # log
            self.pbar_log("train", pbar, epoch, **metrics)

        return metrics

    def close_mosaic(self, epoch: int) -> None:
        if epoch == self.close_mosaic_epoch:
            if hasattr(self.train_loader.dataset, "mosaic"):
                LOGGER.info("Closing dataloader mosaic...")
                self.train_loader.dataset.mosaic = False

    @torch.no_grad()
    def validate(self):
        super().validate()

        # metrics
        metrics = {
            "class": None,
            "images": None,
            "instances": None,
            "vloss": None,
            "precision": None,
            "recall": None,
            "mAP50": None,
            "mAP75": None,
            "mAP50-95": None,
        }
        self.print_metric_titles("val", metrics)

        self.seen = 0
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.conf_thres_val)
        self.metrics = DetMetrics(save_dir=self.cfg["train"]["log_dir"])

        vloss = None

        pbar = tqdm(enumerate(self.val_loader), total=self.val_batches)
        for batch_idx, (imgs, targets) in pbar:
            imgs = imgs.to(self.device, non_blocking=True)
            targets = targets.to(self.device)  # (img_ids, class_ids, bboxes)

            dpreds, preds = self.net(imgs)

            loss, _ = self.criterion(preds=preds, targets=targets, imgs_shape=imgs.shape)
            vloss = (vloss * batch_idx + loss.item()) / (batch_idx + 1) if vloss is not None else loss.item()

            # NMS
            preds = non_max_suppression(
                dpreds,
                conf_thres=self.conf_thres_val,
                iou_thres=self.iou_thres,
                multi_label=True,
                max_det=self.max_det,
                agnostic=self.single_cls,
            )

            scale = torch.tensor([imgs.size(3), imgs.size(2)] * 2, device=self.device)
            self.update_metrics(preds, targets, scale)

            # metrics within the loop
            metrics["vloss"] = vloss
            metrics["class"] = "all"

            if batch_idx == pbar.total - 1:
                self.get_stats()
                self.metrics.confusion_matrix = self.confusion_matrix

                # metrics for final
                metrics["images"] = self.seen
                metrics["vloss"] = vloss
                metrics["instances"] = self.nt_per_class.sum()
                metrics["precision"] = self.metrics.mean_results()[0]
                metrics["recall"] = self.metrics.mean_results()[1]
                metrics["mAP50"] = self.metrics.mean_results()[2]
                metrics["mAP75"] = self.metrics.mean_results()[3]
                metrics["mAP50-95"] = self.metrics.mean_results()[4]

            self.pbar_log("val", pbar, **metrics)

        return metrics

    def _prepare_target(self, si: int, target: torch.Tensor, scale: torch.Tensor):
        """Prepares a batch of images and annotations for validation."""
        idx = target[:, 0] == si
        cls = target[:, 1][idx]
        bbox = target[:, 2:6][idx]
        if len(cls):
            bbox = xywh2xyxy(bbox) * scale  # target boxes
        return {"cls": cls, "bbox": bbox}

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        iou = box_iou(gt_bboxes, detections[:, :4])
        return match_predictions(detections[:, 5], gt_cls, iou, self.iouv)

    def update_metrics(self, preds: torch.Tensor, targets: torch.Tensor, scale: torch.Tensor):
        """Metrics."""
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            ptarget = self._prepare_target(si, targets, scale)
            cls, bbox = ptarget.pop("cls"), ptarget.pop("bbox")
            nl = len(cls)
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # Predictions
            if self.single_cls:
                pred[:, 5] = 0
            predn = self._prepare_pred(pred, scale)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            # Evaluate
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
            if self.plots:
                self.confusion_matrix.process_batch(predn, bbox, cls)
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

    def _prepare_pred(self, pred, scale):
        """Prepares a batch of images and annotations for validation."""
        predn = pred.clone()
        predn[..., :4] *= scale
        return predn

    def get_stats(self):
        """Returns metrics statistics and results dictionary."""
        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}  # to numpy
        self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=self.nc)
        self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=self.nc)
        stats.pop("target_img", None)
        if len(stats) and stats["tp"].any():
            self.metrics.process(**stats)
        return self.metrics.results_dict

    def warmup(self, batch_inters: int, epoch: int) -> int:
        nw = (
            max(round(self.opt_cfg["warmup_epochs"] * self.train_batches), 100)
            if self.opt_cfg["warmup_epochs"] > 0
            else -1
        )  # warmup batches

        if batch_inters <= nw:
            xi = [0, nw]  # x interp
            self.accumulate = max(1, int(np.interp(batch_inters, xi, [1, self.nbs / self.batch_size]).round()))
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

    def criterion(self, preds: Sequence[torch.Tensor], targets: torch.Tensor, imgs_shape: tuple[int]):
        closs = torch.zeros(1, device=self.device)
        bloss = torch.zeros(1, device=self.device)
        dloss = torch.zeros(1, device=self.device)

        strides = torch.tensor([imgs_shape[2] // pred.size(2) for pred in preds], device=self.device)
        pred_distri, pred_scores = torch.cat([xi.view(preds[0].shape[0], self.no, -1) for xi in preds], 2).split(
            (self.reg_max * 4, self.nc), 1
        )  # [bs, no, h1*w1+h2*w2+h3*w3] -> [bs, 4*reg_max, h1*w1+h2*w2+h3*w3] & [bs, nc, h1*w1+h2*w2+h3*w3]

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()  # [bs, h1*w1+h2*w2+h3*w3, nc]
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()  # [bs, h1*w1+h2*w2+h3*w3, 4*reg_max]

        bs = pred_scores.shape[0]
        anchor_points, stride_tensor = make_anchors(preds, strides, 0.5)

        # Targets
        scale_tensor = torch.tensor([imgs_shape[3], imgs_shape[2]] * 2, device=self.device)
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
        closs = self.bce(pred_scores, target_scores.to(pred_scores.dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            bloss, dloss = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        bloss *= self.box_weight
        closs *= self.cls_weight
        dloss *= self.dfl_weight

        tloss = (bloss + closs + dloss) * bs

        loss_component = {"bloss": bloss.item(), "closs": closs.item(), "dloss": dloss.item()}

        return tloss, loss_component

    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor, xywh: bool = False):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=xywh)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor):
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
