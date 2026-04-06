from typing import Any, Mapping, Literal

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, ToTensor, Normalize

from machine_learning.utils.logger import LOGGER
from machine_learning.networks.base import BaseNet
from machine_learning.types.aliases import FilePath
from machine_learning.algorithms.base import AlgorithmBase
from machine_learning.utils.detection import pad_to_square, resize
from machine_learning.utils.segmentation import calculate_miou, visualize_mask, rescale_masks


class PerPixelSegmentation(AlgorithmBase):
    """
    Base class for per-pixel segmentation algorithms, which treat the segmentation task as a pixel-wise classification
    problem. The models is trained to predict the class of each pixel in the input image.

    Note:
        Although per-pixel segmentation can also be used for instance segmentation, as in the classical era dominated by
        Mask R-CNN, where the approach was straightforward: first draw bounding boxes and then perform per-pixel
        segmentation within the local area of these boxes, in the MaskFormer framework, PerPixelSegmentationBase focuses
        on semantic segmentation.
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

        self.loss_weight = self.cfg["algorithm"].get("loss_weight", 1.0)
        self.ignore_value = self.cfg["data"].get("ignore_value", -100)
        self.imgsz = self.cfg["algorithm"]["imgsz"]
        self.single_cls = self.cfg["data"]["single_cls"]
        self.close_mosaic_epoch = self.cfg["algorithm"]["close_mosaic_epoch"]

    def _init_on_trainer(self, train_cfg, dataset):
        """Initialize the datasets, dataloaders, nets, optimizers, and schedulers.
        The attributes that require the dataset parameter are created here.
        """
        super()._init_on_trainer(train_cfg, dataset)

        self.class_names = self.dataset_cfg["class_names"]

    def _init_on_evaluator(self, ckpt, dataset, use_dataset):
        super()._init_on_evaluator(ckpt, dataset, use_dataset)

        self.nc = 1 if self.single_cls else int(self.dataset_cfg["nc"])
        self.class_names = ["object"] if self.single_cls else self.dataset_cfg["class_names"]

    def train_epoch(self, epoch: int, writer: SummaryWriter, log_interval: int = 10) -> dict[str, float]:
        """Train a single epoch."""
        super().train_epoch(epoch, writer, log_interval)

        # close mosaic
        if epoch == int(self.close_mosaic_epoch * self.epochs):
            self.close_dataloader_mosaic()

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

        return metrics, {}

    def close_dataloader_mosaic(self) -> None:
        if hasattr(self.train_loader.dataset, "close_mosaic"):
            LOGGER.info("Closing dataloader mosaic")
            self.train_loader.dataset.close_mosaic()

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Validate after a single train epoch."""
        super().validate()

        # metrics
        total_intersection = None
        total_union = None
        metrics = {"vloss": 0.0, "save_best": 0.0, "batch_miou": 0.0, "miou": 0.0}
        self.print_metric_titles("val", metrics)

        pbar = tqdm(enumerate(self.val_loader), total=self.val_batches)
        for i, batch in pbar:
            # preprocess input and target
            imgs = batch[self.modality].to(self.device, non_blocking=True).float() / 255
            targets = batch["mask"].to(self.device)

            predictions = self.net(imgs)
            loss = self.criterion(predictions, targets)

            # calculate batch mIoU
            batch_miou, (intersection, union) = calculate_miou(predictions, targets, ignore_value=self.ignore_value)
            metrics["batch_miou"] = (metrics["batch_miou"] * i + batch_miou) / (i + 1)

            # accumulate intersection and union for overall mIoU calculation
            if total_intersection is None:
                total_intersection = intersection.clone()
                total_union = union.clone()
            else:
                total_intersection += intersection
                total_union += union

            # add value to val_metrics
            metrics["vloss"] = (metrics["vloss"] * i + loss.item()) / (i + 1)

            if i == self.val_batches - 1:  # calculate overall mIoU at the end of validation
                valid_classes = total_union > 0
                if valid_classes.any():
                    overall_miou = (total_intersection[valid_classes] / total_union[valid_classes]).mean().item()
                else:
                    overall_miou = 0.0
                metrics["miou"] = overall_miou
                metrics["save_best"] = metrics["miou"]

            # log
            self.pbar_log("val", pbar, **metrics)

        return metrics, {}

    def criterion(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate the loss between predictions and targets using cross-entropy loss for per-pixel classification."""
        predictions = F.interpolate(predictions, size=targets.shape[-2:], mode="bilinear", align_corners=False)
        return nn.CrossEntropyLoss(ignore_index=self.ignore_value)(predictions, targets) * self.loss_weight

    @torch.no_grad()
    def eval(
        self,
        img_path: str | FilePath,
        *args,
        **kwargs,
    ) -> None:
        """Evaluate the model effect by visualizing the predicted segmentation mask."""
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
        mask = torch.argmax(preds, dim=1).squeeze(0)
        # rescale masks
        mask = rescale_masks(mask, (self.imgsz, self.imgsz), (h0, w0)).cpu().numpy()

        # visualize results
        visualize_mask(mask=mask)


class MaskSegmentation(AlgorithmBase):
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

            # calculate mIoU
            miou = calculate_miou(predictions, targets)
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
