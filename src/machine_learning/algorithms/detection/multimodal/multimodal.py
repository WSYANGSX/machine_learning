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
from machine_learning.algorithms import YoloV8
from machine_learning.utils.detection import (
    resize,
    non_max_suppression,
    box_iou,
    xywh2xyxy,
    match_predictions,
    ap_per_class,
    pad_to_square,
    visualize_img_bboxes,
    rescale_boxes,
)
from ultralytics.utils.loss import TaskAlignedAssigner, BboxLoss


class MultimodalDetection(YoloV8):
    def __init__(
        self,
        cfg: FilePath | Mapping[str, Any],
        net: BaseNet | None = None,
        name: str | None = None,
        device: Literal["cuda", "cpu", "auto"] = "auto",
        amp: bool = False,
        ema: bool = True,
    ) -> None:
        """
        Implementation of Multimodal object detection algorithm

        Args:
            cfg (FilePath, Mapping[str, Any]): Configuration of the algorithm, it can be yaml file path or cfg dict.
            net (BaseNet): Models required by the Multimodal algorithm.
            name (str): Name of the algorithm.
            device (Literal[&quot;cuda&quot;, &quot;cpu&quot;, &quot;auto&quot;], optional): Running device. Defaults to
            "auto"-automatic selection by algorithm.
            amp (bool): Whether to enable Automatic Mixed Precision. Defaults to False.
            ema (bool): Whether to enable Exponential Moving Average. Defaults to True.
        """
        super().__init__(cfg=cfg, net=net, name=name, device=device, amp=amp, ema=ema)

    def train_epoch(
        self, epoch: int, writer: SummaryWriter, log_interval: int = 10
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Returns training metrics and info dict for the epoch."""
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

        return metrics, {}

    @torch.no_grad()
    def validate(self) -> tuple[dict[str, Any], dict[str, Any]]:
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
        info = {}
        self.print_metric_titles("val", metrics)

        pbar = tqdm(enumerate(self.val_loader), total=self.val_batches)
        for i, batch in pbar:
            imgs = batch["img"].to(self.device, non_blocking=True).float() / 255.0
            irs = batch["ir"].to(self.device, non_blocking=True).float() / 255.0  # convert ir to unit8 in advance
            targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1).to(
                self.device
            )  # (img_ids, class_ids, bboxes)

            net = self.net if not self.ema_enable else self.emas["net"].ema
            preds = net(imgs, irs)
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
                nc=self.nc,
            )  # xyxy [(num_kept_boxes, 6 + num_masks)]*bs

            for si, detection in enumerate(detections):
                metrics["images"] += 1
                # get the original frame of the current image (img_id, class_id, x1, y1, x2, y2)
                pbatch = self.prepare_batch(si, batch)
                tcls, tbox = pbatch.pop("cls"), pbatch.pop("bbox")

                if len(detection) == 0:
                    if len(tcls):
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
                detection = self.prepare_pred(detection, pbatch)
                detection = detection[detection[:, 4].argsort(descending=True)]
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

                    # AP value for each category
                    info["ap_per_class"] = {}
                    for idx, class_idx in enumerate(ap_class):
                        class_name = (
                            self.class_names[class_idx]
                            if hasattr(self, "class_names") and self.class_names
                            else f"Class {class_idx}"
                        )
                        info["ap_per_class"][class_name] = (float(ap50[idx]), float(ap75[idx]), float(ap[idx]))

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

        return metrics, info

    @torch.no_grad()
    def eval(
        self,
        img_path: str | FilePath,
        ir_path: str | FilePath,
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
        bboxes = rescale_boxes(np.array(bboxes.cpu(), dtype=np.float32), self.image_size, w0, h0)
        cls = [int(cid) for cid in cls]

        # visiualization
        visualize_img_bboxes(img0, bboxes, cls, self.class_names)
