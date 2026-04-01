from typing import Any, Mapping, Literal

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter

from machine_learning.networks.base import BaseNet
from machine_learning.types.aliases import FilePath
from machine_learning.algorithms.base import AlgorithmBase


class PerPixelSegmentationBase(AlgorithmBase):
    """
    Base class for per-pixel segmentation algorithms, which treat the segmentation task as a pixel-wise classification
    problem. The models is trained to predict the class of each pixel in the input image.
    """

    def __init__(
        self,
        cfg: FilePath | Mapping[str, Any],
        net: BaseNet | None = None,
        name: str | None = None,
        device: Literal["cuda", "cpu", "auto"] = "auto",
        amp: bool = True,
        ema: bool = True,
        modality: str | None = "img",
    ):
        super().__init__(cfg=cfg, net=net, name=name, device=device, amp=amp, ema=ema)
        self.modality = modality

    def train_epoch(self, epoch: int, writer: SummaryWriter, log_interval: int = 10) -> dict[str, float]:
        """Train a single epoch"""
        super().train_epoch(epoch, writer, log_interval)

        # metrics
        metrics = {"tloss": 0.0}
        self.print_metric_titles("train", metrics)

        pbar = tqdm(enumerate(self.train_loader), total=self.train_batches)
        for i, batch in pbar:
            batch_inters = epoch * self.train_batches + i

            # preprocess input and target
            imgs = batch[self.modality].to(self.device, non_blocking=True).float() / 255
            targets = batch["mask"].to(self.device)

            with autocast(
                device_type=str(self.device), enabled=self.amp
            ):  # Ensure that the autocast scope correctly covers the forward computation
                predictions = self.net(imgs)
                loss = self.criterion(predictions, targets)

            self.backward(loss)
            self.optimizer_step(batch_inters)

            metrics["tloss"] = (metrics["tloss"] * i + loss.item()) / (i + 1)

            if i % log_interval == 0:
                writer.add_scalar("loss/train_batch", loss.item(), batch_inters)  # batch loss

            self.pbar_log("train", pbar, epoch, **metrics)

        return metrics

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Validate after a single train epoch"""
        super().validate()

        # metrics
        metrics = {"vloss": 0.0, "save_best": 0.0, "miou": None}
        self.print_metric_titles("val", metrics)

        pbar = tqdm(enumerate(self.val_loader), total=self.val_batches)
        for i, batch in pbar:
            # preprocess input and target
            imgs = batch[self.modality].to(self.device, non_blocking=True).float() / 255
            targets = batch["mask"].to(self.device)

            predictions = self.net(imgs)
            loss = self.criterion(predictions, targets)

            # calculate mIoU
            miou = self.calculate_miou(predictions, targets)
            metrics["miou"] = (metrics["miou"] * i + miou) / (i + 1)

            # add value to val_metrics
            metrics["vloss"] = (metrics["vloss"] * i + loss.item()) / (i + 1)
            metrics["save_best"] = metrics["miou"]

            # log
            self.pbar_log("val", pbar, **metrics)

        return metrics

    def calculate_miou(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate mean Intersection over Union (mIoU) for the batch."""
        with torch.no_grad():
            # Get predicted class for each pixel
            pred_classes = torch.argmax(predictions, dim=1)

            # Initialize variables to accumulate IoU for each class
            iou_sum = 0.0
            num_classes = predictions.shape[1]

            for c in range(num_classes):
                pred_mask = pred_classes == c
                target_mask = targets == c

                intersection = (pred_mask & target_mask).float().sum()
                union = (pred_mask | target_mask).float().sum()

                if union > 0:
                    iou_sum += intersection / union

            return iou_sum / num_classes if num_classes > 0 else 0.0

    def criterion(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate the loss between predictions and targets using cross-entropy loss for per-pixel classification."""
        fun = nn.CrossEntropyLoss()
        return fun(predictions, targets)

    def eval(
        self,
        img_path: str | FilePath,
        *args,
        **kwargs,
    ) -> None:
        """Evaluate the model effect by visualizing the predicted segmentation mask."""
        self.set_eval()


class MaskSegmentationBase(AlgorithmBase):
    """
    Base class for mask-based segmentation algorithms, which treat the segmentation task as a mask classification and
    prediction problem. The models is trained to predict a set of binary masks and their corresponding class labels for
    regions or objects in the input image.
    """

    def __init__(
        self,
        cfg: FilePath | Mapping[str, Any],
        net: BaseNet | None = None,
        name: str | None = None,
        device: Literal["cuda", "cpu", "auto"] = "auto",
        amp: bool = True,
        ema: bool = True,
        modality: str = "img",
    ):
        super().__init__(cfg=cfg, net=net, name=name, device=device, amp=amp, ema=ema)
        self.modality = modality

    def train_epoch(self, epoch: int, writer: SummaryWriter, log_interval: int = 10) -> dict[str, float]:
        """Train a single epoch"""
        super().train_epoch(epoch, writer, log_interval)

        # metrics
        metrics = {"tloss": 0.0}
        self.print_metric_titles("train", metrics)

        pbar = tqdm(enumerate(self.train_loader), total=self.train_batches)
        for i, batch in pbar:
            batch_inters = epoch * self.train_batches + i

            # preprocess input and target
            data = batch[self.modality].to(self.device, non_blocking=True).float() / 255
            targets = batch["masks"].to(self.device)

            with autocast(
                device_type=str(self.device), enabled=self.amp
            ):  # Ensure that the autocast scope correctly covers the forward computation
                predictions = self.net(data)
                loss = self.criterion(predictions, targets)

            self.backward(loss)
            self.optimizer_step(batch_inters)

            metrics["tloss"] = (metrics["tloss"] * i + loss.item()) / (i + 1)

            if i % log_interval == 0:
                writer.add_scalar("loss/train_batch", loss.item(), batch_inters)  # batch loss

            self.pbar_log("train", pbar, epoch, **metrics)

        return metrics

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Validate after a single train epoch"""
        super().validate()

        # metrics
        metrics = {"vloss": 0.0, "save_best": 0.0, "miou": None}
        self.print_metric_titles("val", metrics)

        pbar = tqdm(enumerate(self.val_loader), total=self.val_batches)
        for i, batch in pbar:
            # preprocess input and target
            imgs = batch[self.modality].to(self.device, non_blocking=True).float() / 255
            targets = batch["masks"].to(self.device)

            predictions = self.net(imgs)
            loss = self.criterion(predictions, targets)

            # calculate mIoU            miou = self.calculate_miou(predictions, targets)
            miou = self.calculate_miou(predictions, targets)
            metrics["miou"] = (metrics["miou"] * i + miou) / (i + 1)

            # add value to val_metrics
            metrics["vloss"] = (metrics["vloss"] * i + loss.item()) / (i + 1)
            metrics["save_best"] = metrics["miou"]

            # log
            self.pbar_log("val", pbar, **metrics)

        return metrics

    def criterion(self, recon: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        """Calculate the loss between predictions and targets."""
        fun = nn.BCEWithLogitsLoss()
        return fun(recon, data)

    def eval(
        self,
        img_path: str | FilePath,
        *args,
        **kwargs,
    ) -> None:
        """Evaluate the model effect by visualizing the predicted segmentation mask."""
        self.set_eval()
