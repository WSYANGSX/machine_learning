from typing import Literal, Mapping, Any, Sequence

import cv2
import math
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, ToTensor, Normalize

from machine_learning.networks import BaseNet
from machine_learning.utils.logger import LOGGER, colorstr
from machine_learning.types.aliases import FilePath
from machine_learning.algorithms.base import AlgorithmBase
from machine_learning.utils.detection import (
    resize,
    non_max_suppression,
    box_iou,
    xywh2xyxy,
    match_predictions,
    ap_per_class,
    pad_to_square,
    visualize_img_bboxes,
    rescale_bboxes,
)
from ultralytics.utils.loss import TaskAlignedAssigner, BboxLoss


class MultimodalDetection(AlgorithmBase):
    def __init__(
        self,
        cfg: FilePath | Mapping[str, Any],
        net: BaseNet | None = None,
        name: str | None = None,
        device: Literal["cuda", "cpu", "auto"] = "auto",
        amp: bool = False,
    ) -> None:
        """
        Implementation of Multimodal object detection algorithm

        Args:
            cfg (FilePath, Mapping[str, Any]): Configuration of the algorithm, it can be yaml file path or cfg dict.
            net (BaseNet): Models required by the Multimodal algorithm.
            name (str): Name of the algorithm. Defaults to "multimodal".
            device (Literal[&quot;cuda&quot;, &quot;cpu&quot;, &quot;auto&quot;], optional): Running device. Defaults to
            "auto"-automatic selection by algorithm.
            amp (bool): Whether to enable Automatic Mixed Precision. Defaults to False.
        """
        super().__init__(cfg=cfg, net=net, name=name, device=device, amp=amp)

        # main parameters of the algorithm
        self.task = self.cfg["algorithm"]["task"]
        self.imgsz = self.cfg["algorithm"]["imgsz"]
        self.reg_max = self.cfg["algorithm"]["reg_max"]
        self.use_dfl = self.reg_max > 1
        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=self.device)
        self.close_mosaic_epoch = self.cfg["algorithm"]["close_mosaic_epoch"]
        self.max_det = self.cfg["algorithm"]["max_det"]
        self.single_cls = self.cfg["data"]["single_cls"]
        self.plot = self.cfg["algorithm"].get("plot", False)

        # threshold
        self.iou_thres = self.cfg["algorithm"]["iou_thres"]
        self.conf_thres_val = self.cfg["algorithm"]["conf_thres_val"]
        self.conf_thres_det = self.cfg["algorithm"]["conf_thres_det"]

        # weight
        self.box_weight = self.cfg["algorithm"].get("box")
        self.cls_weight = self.cfg["algorithm"].get("cls")
        self.dfl_weight = self.cfg["algorithm"].get("dfl")

        self.bbox_loss = BboxLoss(self.reg_max).to(self.device)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()

    def _init_on_trainer(self, train_cfg, dataset):
        """Initialize the datasets, dataloaders, nets, optimizers, and schedulers.
        The attributes that require the dataset parameter are created here
        """
        super()._init_on_trainer(train_cfg, dataset)

        self.topk = self.cfg["algorithm"]["topk"]
        self.alpha = self.cfg["algorithm"]["alpha"]
        self.beta = self.cfg["algorithm"]["beta"]
        self.nc = self.dataset_cfg["nc"]
        self.class_names = self.dataset_cfg["class_names"]

        self.assigner = TaskAlignedAssigner(
            topk=self.topk, num_classes=self.dataset_cfg["nc"], alpha=self.alpha, beta=self.beta
        )

    def _init_on_evaluator(self, ckpt, dataset, use_dataset):
        super()._init_on_evaluator(ckpt, dataset, use_dataset)

        self.nc = 1 if self.single_cls else int(self.dataset_cfg["nc"])
        self.class_names = self.dataset_cfg["class_names"]

    def _init_optimizers(self) -> None:
        self.opt_cfg = self._cfg["optimizer"]

        weight_decay = self.opt_cfg.get("weight_decay", 1e-5) * self.batch_size * self.accumulate / self.nbs
        iterations = (
            math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.nbs)) * self.trainer_cfg["epochs"]
        )
        self.optimizer = self.build_optimizer(
            name=self.opt_cfg.get("opt", "auto"),
            lr=self.opt_cfg.get("lr", 0.001),
            momentum=self.opt_cfg.get("momentum", 0.9),
            decay=weight_decay,
            iterations=iterations,
        )

        self._add_optimizer("optimizer", self.optimizer)

    def _init_schedulers(self) -> None:
        self.sch_config = self._cfg["scheduler"]

        if self.sch_config.get("sched") == "CustomLRDecay":
            self.lf = (
                lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.opt_cfg["final_factor"])
                + self.opt_cfg["final_factor"]
            )  # linear
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
            self._add_scheduler("scheduler", self.scheduler)

        else:
            LOGGER.warning(f"Unknown scheduler type '{self.sch_config.get('type')}', no scheduler configured.")

    def train_epoch(self, epoch: int, writer: SummaryWriter, log_interval: int = 10) -> dict[str, float]:
        super().train_epoch(epoch, writer, log_interval)

        # close mosaic
        if epoch == int(self.close_mosaic_epoch * self.epochs):
            self.close_dataloader_mosaic()

        # log metrics
        metrics = {"tloss": 0.0, "bloss": 0.0, "dloss": 0.0, "closs": 0.0, "instances": 0, "img_size": None}
        self.print_metric_titles("train", metrics)

        pbar = tqdm(enumerate(self.train_loader), total=self.train_batches)
        for i, batch in pbar:
            # Warmup
            batches = epoch * self.train_batches + i
            self.warmup(batches, epoch)

            # load data
            imgs = batch["img"].to(self.device, non_blocking=True).float() / 255.0
            irs = batch["ir"].to(self.device, non_blocking=True).float() / 255.0  # convert ir to unit8 in advance
            targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1).to(
                self.device
            )  # (img_ids, class_ids, bboxes)

            with autocast(
                device_type=str(self.device), enabled=self.amp
            ):  # Ensure that the autocast scope correctly covers the forward computation
                preds = self.net(imgs, irs)
                loss, lc = self.criterion(preds=preds, targets=targets, imgs_shape=imgs.shape)

            # Gradient backpropagation
            self.backward(loss)
            # Parameter optimization
            self.optimizer_step(batches)

            # Metrics
            bloss, closs, dloss = lc["bloss"], lc["closs"], lc["dloss"]  # component loss

            metrics["tloss"] = (metrics["tloss"] * i + loss.item()) / (i + 1)  # tloss
            metrics["bloss"] = (metrics["bloss"] * i + bloss) / (i + 1)
            metrics["closs"] = (metrics["closs"] * i + closs) / (i + 1)
            metrics["dloss"] = (metrics["dloss"] * i + dloss) / (i + 1)
            metrics["img_size"] = imgs.size(2)
            metrics["instances"] = targets.size(0)

            if i % log_interval == 0:
                writer.add_scalar("bloss/train_batch", bloss, batches)
                writer.add_scalar("closs/train_batch", closs, batches)
                writer.add_scalar("dloss/train_batch", dloss, batches)

            # log
            self.pbar_log("train", pbar, epoch, **metrics)

        return metrics

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        super().validate()

        # log metrics
        stats = []
        metrics = {
            "class": "all",
            "images": 0,
            "vloss": 0.0,
            "save_best": 0.0,
            "labels": 0,
            "precision": None,
            "recall": None,
            "mAP.5": None,
            "mAP.75": None,
            "mAP.5-.95": None,
        }
        self.print_metric_titles("val", metrics)

        pbar = tqdm(enumerate(self.val_loader), total=self.val_batches)
        for i, batch in pbar:
            imgs = batch["img"].to(self.device, non_blocking=True).float() / 255.0
            irs = batch["ir"].to(self.device, non_blocking=True).float() / 255.0  # convert ir to unit8 in advance
            targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1).to(
                self.device
            )  # (img_ids, class_ids, bboxes)

            preds = self.net(imgs, irs)
            loss, _ = self.criterion(preds=preds, targets=targets, imgs_shape=imgs.shape)

            # metrics
            metrics["save_best"] = metrics["vloss"] = (metrics["vloss"] * i + loss.item()) / (i + 1)

            # prepare preds
            detections = self.decode_preds(preds, imgs.size(2))  # [bs, sum(h*w), 4 + nc]
            # NMS: Fewer overlapping boxes
            detections = non_max_suppression(
                detections.permute(0, 2, 1),
                conf_thres=self.conf_thres_val,
                iou_thres=self.iou_thres,
                multi_label=True,
                max_det=self.max_det,
                agnostic=self.single_cls,
            )  # xyxy [(num_kept_boxes, 6 + num_masks)]*bs

            # prepare targets
            scale = torch.tensor([imgs.shape[3], imgs.shape[2]] * 2, device=self.device)
            targets_abs = targets.clone()
            bbox_abs = targets[:, 2:6] * scale  # convert to abs xywh
            targets_abs[:, 2:6] = xywh2xyxy(bbox_abs)  # convert to xyxy (img_ids, class_ids, bboxes)

            for si, detection in enumerate(detections):
                metrics["images"] += 1
                # get the real frame of the current image (img_id, class_id, x1, y1, x2, y2)
                labels = targets_abs[targets_abs[:, 0] == si, 1:]

                # extract the category and coordinates of the real box
                tcls = labels[:, 0] if len(labels) > 0 else torch.empty(0, device=self.device)
                tbox = labels[:, 1:5] if len(labels) > 0 else torch.empty(0, 4, device=self.device)

                if len(detection) == 0:
                    if len(labels):
                        stats.append(
                            (
                                torch.zeros(0, self.niou, dtype=torch.bool, device=self.device),  # TP
                                torch.zeros(0, device=self.device),  # confidence scores
                                torch.zeros(0, device=self.device),  # pred_classes
                                tcls,  # ground truth classes
                            )
                        )
                    continue

                # Prediction box format: [x1, y1, x2, y2, conf, class]
                pred_boxes = detection[:, :4]
                pred_scores = detection[:, 4]
                pred_classes = detection[:, 5]

                # calculate IoU
                iou = box_iou(tbox, pred_boxes)  # shape: [n_gt, n_pred]
                # Determine whether each prediction box is regarded as a correct detection under each IoU threshold
                tp = match_predictions(pred_classes, tcls, iou, self.iouv)

                # record statistical information
                stats.append((tp, pred_scores, pred_classes, tcls))

            if i == self.val_batches - 1:
                # calculate mAP metrics
                stats = [torch.cat(x, 0) for x in zip(*stats)]  # to torch

                if len(stats) and stats[0].any():
                    p, r, ap, f1, ap_class = ap_per_class(
                        *stats, plot=self.plot, save_dir=self.trainer_cfg["log_dir"], names=self.class_names
                    )
                    ap50, ap75, ap = ap[:, 0], ap[:, 5], ap.mean(1)  # AP@0.5, AP@0.75, AP@0.5:0.95
                    mp, mr, map50, map75, map = p.mean(), r.mean(), ap50.mean(), ap75.mean(), ap.mean()
                    nt = np.bincount(
                        stats[3].cpu().numpy().astype(np.int64), minlength=self.nc
                    )  # number of targets per class
                    metrics["mAP.5"] = map50
                    metrics["mAP.75"] = map75
                    metrics["mAP.5-.95"] = map
                    metrics["precision"] = mp
                    metrics["recall"] = mr

                else:
                    nt = np.zeros(1, dtype=np.uint8)

                metrics["labels"] = nt.sum()

            self.pbar_log("val", pbar, **metrics)

        return metrics

    def decode_preds(self, preds: torch.Tensor, img_size: int) -> torch.Tensor:
        # decode preds
        x = []
        for pred in preds:
            bs, _, h, w = pred.shape
            pred = pred.permute(0, 2, 3, 1).reshape(bs, h * w, -1)
            x.append(pred)
        x = torch.cat(x, 1)  # [bs， sum(h*w), nc + 4*reg_max]
        # Separate the bounding box from the category score
        box, cls = x.split((self.reg_max * 4, self.nc), 2)

        strides = torch.tensor([img_size // pred.size(2) for pred in preds], device=self.device)
        anchor_points, stride_tensor = make_anchors(preds, strides, 0.5)
        dbox = self.bbox_decode(anchor_points, box, xywh=True) * stride_tensor
        detections = torch.cat((dbox, cls.sigmoid()), 2)

        return detections  # [bs, sum(h*w), 4 + nc]

    def criterion(self, preds: Sequence[torch.Tensor], targets: torch.Tensor, imgs_shape: tuple[int]):
        closs = torch.zeros(1, device=self.device)
        bloss = torch.zeros(1, device=self.device)
        dloss = torch.zeros(1, device=self.device)

        strides = torch.tensor([imgs_shape[2] // pred.size(2) for pred in preds], device=self.device)
        pred_distri, pred_scores = torch.cat(
            [xi.view(preds[0].shape[0], self.nc + self.reg_max * 4, -1) for xi in preds], 2
        ).split(
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

        target_scores_sum = target_scores.sum().clamp(min=1.0)

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

        loss = (bloss + closs + dloss) * bs
        loss_component = {"bloss": bloss.item(), "closs": closs.item(), "dloss": dloss.item()}

        return loss, loss_component

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

    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor, xywh: bool = False):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=xywh)

    @torch.no_grad()
    def eval(
        self,
        img_path: str | FilePath,
        conf_thres: float | None = None,
        iou_thres: float | None = None,
        *args,
        **kwargs,
    ) -> None:
        self.set_eval()

        # read image
        img0 = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        h0, w0, _ = img0.shape

        # scale to square
        padded_img = pad_to_square(img=img0, pad_values=0)

        # to tensor / normalize
        tfs = Compose([ToTensor(), Normalize(mean=[0, 0, 0], std=[1, 1, 1])])
        img = tfs(padded_img)
        img = resize(img, size=self.imgsz).unsqueeze(0).to(self.device)

        # input image to model
        preds = self.net(img)

        # decode preds
        detections = self.decode_preds(preds, img.size(2))  # xywh
        # NMS
        detections = non_max_suppression(
            detections.permute(0, 2, 1),
            conf_thres=conf_thres if conf_thres is not None else self.conf_thres_det,
            iou_thres=iou_thres if iou_thres is not None else self.iou_thres,
            multi_label=True,
            max_det=self.max_det,
            agnostic=self.single_cls,
        )  # xyxy

        # rescale to img coordiante
        detection = detections[0]

        bboxes, conf, cls = detection.split((4, 1, 1), dim=1)
        bboxes = rescale_bboxes(np.array(bboxes.cpu(), dtype=np.float32), self.image_size, w0, h0)
        cls = [int(cid) for cid in cls]

        # visiualization
        visualize_img_bboxes(img0, bboxes, cls, self.class_names)

    def warmup(self, batch_inters: int, epoch: int) -> int:
        wb = (
            max(round(self.opt_cfg["warmup_epochs"] * self.train_batches), 100)
            if self.opt_cfg["warmup_epochs"] > 0
            else -1
        )  # warmup batches

        if batch_inters <= wb:
            xi = [0, wb]  # x interp
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

    def build_optimizer(self, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """
        Constructs an optimizer for the given model, based on the specified optimizer name, learning rate, momentum,
        weight decay, and number of iterations.

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected
                based on the number of iterations. Default: 'auto'.
            lr (float, optional): The learning rate for the optimizer. Default: 0.001.
            momentum (float, optional): The momentum factor for the optimizer. Default: 0.9.
            decay (float, optional): The weight decay for the optimizer. Default: 1e-5.
            iterations (float, optional): The number of iterations, which determines the optimizer if
                name is 'auto'. Default: 1e5.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        """
        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        if name == "auto":
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                f"ignoring 'lr={self.opt_cfg['lr']}' and 'momentum={self.opt_cfg['momentum']}' and "
                f"determining best 'optimizer', 'lr' and 'momentum' automatically... "
            )
            lr_fit = round(0.002 * 5 / (4 + self.dataset_cfg["nc"]), 6)  # lr0 fit equation to 6 decimal places
            name, lr, momentum = ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            self.opt_cfg["warmup_bias_lr"] = 0.0  # no higher than 0.01 for Adam

        for module_name, module in self.net.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn):  # weight (no decay)
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

        optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "auto"}
        name = {x.lower(): x for x in optimizers}.get(name.lower())
        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optimizer = getattr(torch.optim, name, torch.optim.Adam)(
                g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0
            )
        elif name == "RMSProp":
            optimizer = torch.optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == "SGD":
            optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers {optimizers}. "
                "Request support for addition optimizers at https://github.com/ultralytics/ultralytics."
            )

        optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f"{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)"
        )

        return optimizer


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
