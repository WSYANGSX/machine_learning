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
from machine_learning.types.aliases import FilePath
from machine_learning.utils.logger import LOGGER, colorstr
from machine_learning.algorithms.base import AlgorithmBase
from machine_learning.utils.detection import (
    resize,
    box_iou,
    non_max_suppression,
    xywh2xyxy,
    pad_to_square,
    visualize_img_bboxes,
    rescale_bboxes,
    match_predictions,
)
from ultralytics.utils.loss import TaskAlignedAssigner, BboxLoss
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics
from ultralytics.utils.ops import scale_boxes


class YoloV13(AlgorithmBase):
    def __init__(
        self,
        cfg: FilePath | Mapping[str, Any],
        net: BaseNet | None = None,
        name: str | None = "yolo_v13",
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
            name (str): Name of the algorithm. Defaults to "yolo_v13".
            device (Literal[&quot;cuda&quot;, &quot;cpu&quot;, &quot;auto&quot;], optional): Running device. Defaults to
            "auto"-automatic selection by algorithm.
            amp (bool): Whether to enable Automatic Mixed Precision. Defaults to False.
        """
        super().__init__(cfg=cfg, net=net, name=name, device=device, amp=amp)

        # main parameters of the algorithm
        self.task = self.cfg["algorithm"]["task"]
        self.image_size = self.cfg["algorithm"]["imgsz"]
        self.reg_max = self.cfg["algorithm"]["reg_max"]
        self.use_dfl = self.reg_max > 1
        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=self.device)
        self.close_mosaic_epoch = self.cfg["algorithm"]["close_mosaic_epoch"]
        self.max_det = self.cfg["algorithm"]["max_det"]
        self.single_cls = self.cfg["data"]["single_cls"]

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

        # IoU vector for mAP@0.5:0.95
        self.iouv = torch.linspace(0.5, 0.95, 10)
        self.niou = self.iouv.numel()
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
        self.plots = self.cfg["algorithm"].get("plot", False)

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

        self.assigner = TaskAlignedAssigner(topk=self.topk, num_classes=self.nc, alpha=self.alpha, beta=self.beta)
        self.metrics = DetMetrics(save_dir=self.trainer_cfg["log_dir"])
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.conf_thres_val)

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
        self.scheduler = None

        if self.sch_config.get("sched") == "CustomLRDecay":
            self.lf = (
                lambda x: max(1 - x / self.trainer_cfg["epochs"], 0) * (1.0 - self.opt_cfg["final_factor"])
                + self.opt_cfg["final_factor"]
            )  # linear
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
            self._add_scheduler("scheduler", self.scheduler)

        else:
            LOGGER.warning(f"Unknown scheduler type '{self.sch_config.get('type')}', no scheduler configured.")

    def train_epoch(self, epoch: int, writer: SummaryWriter, log_interval: int = 10) -> dict[str, float]:
        super().train_epoch(epoch, writer, log_interval)

        # close mosaic
        if epoch == int(self.close_mosaic_epoch * self.trainer_cfg["epochs"]):
            self.close_dataloader_mosaic()

        # log metrics
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
        for i, batch in pbar:
            # Warmup
            batch_inters = epoch * self.train_batches + i
            self.warmup(batch_inters, epoch)

            # Load data
            imgs = batch["img"].to(self.device, non_blocking=True).float() / 255
            targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1).to(
                self.device
            )  # (img_ids, class_ids, bboxes)

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
            tloss = (tloss * i + loss.item()) / (i + 1) if tloss is not None else loss.item()
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

            if i % log_interval == 0:
                writer.add_scalar("bloss/train_batch", bloss, batch_inters)
                writer.add_scalar("closs/train_batch", closs, batch_inters)
                writer.add_scalar("dloss/train_batch", dloss, batch_inters)

            # log
            self.pbar_log("train", pbar, epoch, **metrics)

        return metrics

    def close_dataloader_mosaic(self) -> None:
        if hasattr(self.train_loader.dataset, "close_mosaic"):
            LOGGER.info("Closing dataloader mosaic")
            self.train_loader.dataset.close_mosaic()

    @torch.no_grad()
    def validate(self):
        super().validate()

        # log metrics
        metrics = {
            "class": "all",
            "images": None,
            "vloss": None,
            "sloss": None,
            "precision": None,
            "recall": None,
            "mAP50": None,
            "mAP75": None,
            "mAP50-95": None,
        }
        self.print_metric_titles("val", metrics)

        vloss = None
        self.seen = 0

        pbar = tqdm(enumerate(self.val_loader), total=self.val_batches)
        for batch_idx, batch in pbar:
            batch["img"] = batch["img"].to(self.device, non_blocking=True)
            for k in ["batch_idx", "cls", "bboxes"]:
                batch[k] = batch[k].to(self.device)
            imgs = batch["img"].float() / 255
            # (img_ids, class_ids, bboxes)
            targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)

            preds = self.net(imgs)
            loss, _ = self.criterion(preds=preds, targets=targets, imgs_shape=imgs.shape)
            vloss = (vloss * batch_idx + loss.item()) / (batch_idx + 1) if vloss is not None else loss.item()
            metrics["vloss"] = metrics["sloss"] = vloss

            detections = self.decode_preds(preds, imgs.size(2))
            detections = non_max_suppression(
                detections.permute(0, 2, 1),
                conf_thres=self.conf_thres_val,
                iou_thres=self.iou_thres,
                multi_label=True,
                max_det=self.max_det,
                agnostic=self.single_cls,
            )

            self.update_metrics(detections, batch)

            if batch_idx == self.val_batches - 1:
                states = self.get_stats()
                metrics["precision"] = states["metrics/precision(B)"]
                metrics["recall"] = states["metrics/recall(B)"]
                metrics["mAP50"] = states["metrics/mAP50(B)"]
                metrics["mAP75"] = states["metrics/mAP75(B)"]
                metrics["mAP50-95"] = states["metrics/mAP50-95(B)"]

            metrics["images"] = self.seen

            self.pbar_log("val", pbar, **metrics)

        return metrics

    def update_metrics(self, detections, batch):
        """Metrics."""
        for si, detection in enumerate(detections):
            self.seen += 1
            npr = len(detection)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
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
                detection[:, 5] = 0
            predn = self._prepare_pred(detection, pbatch)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            # Evaluate
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
            if self.plots:
                self.confusion_matrix.process_batch(predn, bbox, cls)
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

    def _prepare_batch(self, si, batch):
        """Prepares a batch of images and annotations for validation."""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox = xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # target boxes
            scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)  # native-space labels
        return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}

    def _prepare_pred(self, pred, pbatch):
        """Prepares a batch of images and annotations for validation."""
        predn = pred.clone()
        # native-space pred
        scale_boxes(pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"])
        return predn

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Return correct prediction matrix.
        """
        iou = box_iou(gt_bboxes, detections[:, :4])
        return match_predictions(detections[:, 5], gt_cls, iou, self.iouv)

    def get_stats(self):
        """Returns metrics statistics and results dictionary."""
        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}  # to numpy
        self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=self.nc)
        self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=self.nc)
        stats.pop("target_img", None)
        if len(stats) and stats["tp"].any():
            self.metrics.process(**stats)
        return self.metrics.results_dict

    def decode_preds(self, preds: torch.Tensor, img_size: int) -> list[torch.Tensor]:
        # decode preds
        x = []
        for pred in preds:
            bs, _, h, w = pred.shape
            pred = pred.permute(0, 2, 3, 1).reshape(bs, h * w, -1)
            x.append(pred)
        x = torch.cat(x, 1)
        # Separate the bounding box from the category score
        box, cls = x.split((self.reg_max * 4, self.nc), 2)

        strides = torch.tensor([img_size // pred.size(2) for pred in preds], device=self.device)
        anchor_points, stride_tensor = make_anchors(preds, strides, 0.5)
        dbox = self.bbox_decode(anchor_points, box, xywh=True) * stride_tensor
        predictions = torch.cat((dbox, cls.sigmoid()), 2)

        return predictions

    def criterion(self, preds: Sequence[torch.Tensor], targets: torch.Tensor, imgs_shape: tuple[int]):
        closs = torch.zeros(1, device=self.device)
        bloss = torch.zeros(1, device=self.device)
        dloss = torch.zeros(1, device=self.device)

        strides = torch.tensor([imgs_shape[2] // pred.size(2) for pred in preds], device=self.device)
        pred_distri, pred_scores = torch.cat(
            [xi.view(preds[0].shape[0], self.dataset_cfg["nc"] + self.reg_max * 4, -1) for xi in preds], 2
        ).split(
            (self.reg_max * 4, self.dataset_cfg["nc"]), 1
        )  # [bs, no, h1*w1+h2*w2+h3*w3] -> [bs, 4*reg_max, h1*w1+h2*w2+h3*w3] & [bs, nc, h1*w1+h2*w2+h3*w3]

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()  # [bs, h1*w1+h2*w2+h3*w3, nc]
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()  # [bs, h1*w1+h2*w2+h3*w3, 4*reg_max]
        if torch.isnan(pred_scores).any() or torch.isnan(pred_distri).any():
            raise ValueError("The output of model contain Nan value!")

        bs = pred_scores.shape[0]
        anchor_points, stride_tensor = make_anchors(preds, strides, 0.5)

        # Targets
        scale_tensor = torch.tensor([imgs_shape[3], imgs_shape[2]] * 2, device=self.device)
        targets = self.target_preprocess(targets, bs, scale_tensor=scale_tensor)
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

    def target_preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor):
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
        img = resize(img, size=self.image_size).unsqueeze(0).to(self.device)

        # input image to model
        preds = self.net(img)

        predictions = self.decode_preds(preds, img.size(2))

        # NMS
        preds = non_max_suppression(
            predictions.permute(0, 2, 1),
            conf_thres=conf_thres if conf_thres is not None else self.conf_thres_det,
            iou_thres=iou_thres if iou_thres is not None else self.iou_thres,
            multi_label=True,
            max_det=self.max_det,
            agnostic=self.single_cls,
        )

        # rescale to img coordiante
        pred = preds[0]

        bboxes, conf, cls = pred.split((4, 1, 1), dim=1)
        bboxes = rescale_bboxes(np.array(bboxes.cpu(), dtype=np.float32), self.image_size, w0, h0)
        cls = [int(cid) for cid in cls]

        # visiualization
        visualize_img_bboxes(img0, bboxes, cls, self.class_names)

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
