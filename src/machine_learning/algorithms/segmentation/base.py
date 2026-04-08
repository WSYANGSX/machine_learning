from typing import Any, Mapping, Literal

import cv2
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, ToTensor, Normalize
from machine_learning.utils.logger import LOGGER
from machine_learning.networks.base import BaseNet
from machine_learning.types.aliases import FilePath
from machine_learning.algorithms.base import AlgorithmBase
from machine_learning.utils.streams import VideoStream, WebcamStream
from machine_learning.utils.detection import pad_to_square, resize
from machine_learning.utils.segmentation import calculate_miou, colour_mask, rescale_masks


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

    def on_epoch_start(self, epoch: int):
        # close mosaic
        if epoch == int(self.close_mosaic_epoch * self.epochs):
            self.close_dataloader_mosaic()

    def _init_metrics(self, mode: Literal["train", "val", "test"]) -> dict[str, Any]:
        if mode == "train":
            return {"tloss": 0.0}
        elif mode == "val":
            return {"vloss": 0.0, "best_fitness": 0.0, "batch_miou": 0.0, "miou": 0.0}
        else:
            return {"miou": 0.0}

    def _prepare_batch(self, batch: dict[str, Any], mode: Literal["train", "val", "test"]) -> dict[str, Any]:
        data = {}
        data["imgs"] = batch[self.modality].to(self.device, non_blocking=True).float() / 255
        data["targets"] = batch["mask"].to(self.device)
        return data

    def _forward_batch(
        self, net: BaseNet, data: dict[str, Any], mode: Literal["train", "val", "test"]
    ) -> dict[str, Any]:
        if mode in ("train", "val"):
            imgs = data["imgs"]
            targets = data["targets"]

            # Loss calculation
            preds = net(imgs)
            loss = self.criterion(preds, targets)

            return {"loss": loss, "preds": preds}
        else:
            imgs = data["imgs"]
            preds = net(imgs)

            return {"preds": preds}

    def _post_process(
        self,
        batch_idx: int,
        res: dict[str, Any],
        batch: dict[str, Any],
        data: dict[str, Any],
        metrics: dict[str, Any],
        statistics: list,
        info: dict[str, Any],
        mode: Literal["train", "val", "test"],
    ) -> None:
        if mode == "train":
            loss = res["loss"]
            metrics["tloss"] = (metrics["tloss"] * batch_idx + loss.item()) / (batch_idx + 1)

        elif mode == "val":
            loss, preds = res["loss"], res["preds"]
            metrics["vloss"] = (metrics["vloss"] * batch_idx + loss.item()) / (batch_idx + 1)

            targets = data["targets"]
            batch_miou, (intersection, union) = calculate_miou(preds, targets, ignore_value=self.ignore_value)
            metrics["batch_miou"] = (metrics["batch_miou"] * batch_idx + batch_miou) / (batch_idx + 1)

            # accumulate intersection and union for overall mIoU calculation
            if len(statistics) == 0:
                statistics.append(intersection.detach().clone())  # detach(), defensive programming, optional
                statistics.append(union.detach().clone())  # [intersection, union]
            else:
                statistics[0] += intersection
                statistics[1] += union

            if batch_idx == self.val_batches - 1:  # calculate overall mIoU at the end of validation
                valid_classes = statistics[1] > 0
                if valid_classes.any():
                    overall_miou = (statistics[0][valid_classes] / statistics[1][valid_classes]).mean().item()
                else:
                    overall_miou = 0.0
                metrics["miou"] = overall_miou
                metrics["best_fitness"] = metrics["miou"]

        else:
            preds = res["preds"]
            targets = data["targets"]
            _, (intersection, union) = calculate_miou(preds, targets, ignore_value=self.ignore_value)

            # accumulate intersection and union for overall mIoU calculation
            if len(statistics) == 0:
                statistics.append(intersection.detach().clone())
                statistics.append(union.detach().clone())  # [intersection, union]
            else:
                statistics[0] += intersection
                statistics[1] += union

            if batch_idx == self.test_batches - 1:  # calculate overall mIoU at the end of validation
                valid_classes = statistics[1] > 0
                if valid_classes.any():
                    overall_miou = (statistics[0][valid_classes] / statistics[1][valid_classes]).mean().item()
                else:
                    overall_miou = 0.0
                metrics["miou"] = overall_miou

    def write(self, batches: int, writer: SummaryWriter, res: dict[str, Any], metrics: dict[str, Any]) -> None:
        writer.add_scalar("loss/train_batch", metrics["tloss"], batches)  # batch loss

    def close_dataloader_mosaic(self) -> None:
        if hasattr(self.train_loader.dataset, "close_mosaic"):
            LOGGER.info("Closing dataloader mosaic")
            self.train_loader.dataset.close_mosaic()

    def criterion(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate the loss between predictions and targets using cross-entropy loss for per-pixel classification."""
        predictions = F.interpolate(predictions, size=targets.shape[-2:], mode="bilinear", align_corners=False)
        return nn.CrossEntropyLoss(ignore_index=self.ignore_value)(predictions, targets) * self.loss_weight

    # TODO
    @torch.no_grad()
    def predict(self, stream: str | VideoStream | WebcamStream, *args, **kwargs) -> None:
        """Make predictions from different special data stream."""
        super().predict(stream, *args, **kwargs)

        if isinstance(stream, (str, FilePath)):
            self._predict_single_frame(stream, *args, **kwargs)

        elif isinstance(stream, (VideoStream, WebcamStream)):
            self._predict_stream(stream, *args, **kwargs)

    def _predict_single_frame(self, img_path: FilePath, *args, **kwargs):
        """Evaluate the single-frame image."""
        # read image
        img0 = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img0 is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)  # BGR -> RGB

        res_img = self._inference_and_preparation(img0, *args, **kwargs)

        res_img_bgr = cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR)
        cv2.imshow("Detection result", res_img_bgr)
        cv2.waitKey(0)  # Wait for a key press
        cv2.destroyAllWindows()

    def _predict_stream(self, stream: VideoStream | WebcamStream, *args, **kwargs):
        """Evaluate video images or live data streams."""
        # Calculate the exact inter-frame delay required for offline video
        # The camera is internally limited, so delay=1
        is_video = isinstance(stream, VideoStream)
        delay = max(1, int(1000 / stream.fps)) if is_video and stream.fps > 0 else 1

        LOGGER.info("Starting stream evaluation. Press 'q' to stop.")

        for frames in stream:
            if isinstance(frames, dict):
                frame = frames.get(self.modality)
                if frame is None:
                    raise ValueError(f"Stream does not contain required modality: {self.modality}")

            else:
                frame = frames

            f_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res_rgb = self._inference_and_preparation(f_rgb, *args, **kwargs)
            show_frame = cv2.cvtColor(res_rgb, cv2.COLOR_RGB2BGR)

            cv2.imshow("Detection result", show_frame)

            if cv2.waitKey(delay) & 0xFF == ord("q"):
                LOGGER.info("Stream stopped by user.")
                break

        cv2.destroyAllWindows()

    def _inference_and_preparation(self, img0: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Core reasoning and rendering logic."""
        # read image
        h0, w0, _ = img0.shape

        # scale to square
        padded_img = pad_to_square(img=img0, pad_values=(114, 114, 114))

        # to tensor / normalize
        tfs = Compose([ToTensor(), Normalize(mean=[0, 0, 0], std=[1, 1, 1])])
        img = tfs(padded_img)
        img = resize(img, size=self.imgsz).unsqueeze(0).to(self.device)

        # input image to model
        preds = self.net(img)
        mask = torch.argmax(preds, dim=1).squeeze(0)
        # rescale masks
        mask = rescale_masks(mask, (self.imgsz, self.imgsz), (h0, w0)).cpu().numpy()
        mask, num_items = colour_mask(mask)


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
