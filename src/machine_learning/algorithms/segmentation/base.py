from typing import Any, Mapping, Literal

import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from machine_learning.utils.logger import LOGGER
from machine_learning.networks.base import BaseNet
from machine_learning.types.aliases import FilePath
from machine_learning.algorithms.base import AlgorithmBase
from torchvision.transforms import Compose, ToTensor, Normalize
from machine_learning.utils.streams import VideoStream, WebcamStream
from machine_learning.utils.detect import pad_to_square, resize
from machine_learning.utils.segment import colour_mask, rescale_masks, generate_gt_edges, SegmentMetrics


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

        # loss weight
        self.ce_weight = self.cfg["algorithm"].get("ce_weight")
        self.geo_weight = self.cfg["algorithm"].get("geo_weight")

        self.imgsz = self.cfg["algorithm"]["imgsz"]
        self.single_cls = self.cfg["data"]["single_cls"]
        self.close_mosaic_epoch = self.cfg["algorithm"]["close_mosaic_epoch"]

    def _init_on_trainer(
        self,
        train_cfg: dict[str, Any],
        dataset: str | Mapping[str, Any],
    ):
        """Initialize the datasets, dataloaders, nets, optimizers, and schedulers.
        The attributes that require the dataset parameter are created here.
        """
        super()._init_on_trainer(train_cfg, dataset)

        self.nc = 2 if self.single_cls else int(self.dataset_cfg["nc"])
        self.class_names = ["object"] if self.single_cls else self.dataset_cfg["class_names"]
        self.ignore_value = self.dataset_cfg.get("ignore_value", -100)
        self.metrics = SegmentMetrics(nc=self.nc)

    def _init_on_evaluator(
        self,
        ckpt: str,
        dataset: str | Mapping[str, Any] | None = None,
        load_dataset: bool = True,
        plot: bool | None = False,
        save_dir: str | None = None,
    ):
        super()._init_on_evaluator(ckpt, dataset, load_dataset, plot, save_dir)

        self.nc = 2 if self.single_cls else int(self.dataset_cfg["nc"])
        self.class_names = ["object"] if self.single_cls else self.dataset_cfg["class_names"]
        self.ignore_value = self.dataset_cfg.get("ignore_value", -100)
        self.metrics = SegmentMetrics(nc=self.nc)

    def on_epoch_start(self, epoch: int):
        # close mosaic
        if epoch == int(self.close_mosaic_epoch * self.epochs):
            self.close_dataloader_mosaic()

    def _init_metrics(self, mode: Literal["train", "val", "test"]) -> dict[str, Any]:
        if mode == "train":
            return (
                {"tloss": 0.0, "img_size": None}
                if not self.geo_weight
                else {"tloss": 0.0, "ce_loss": 0.0, "geo_loss": 0.0, "img_size": None}
            )
        elif mode == "val":
            return {
                "vloss": 0.0,
                "best_fitness": 0.0,
                "miou": 0.0,
                "overall acc": 0.0,
                "mean acc": 0.0,
                "freqW acc": 0.0,
            }
        else:
            return {"miou": 0.0, "overall acc": 0.0, "mean acc": 0.0, "freqW acc": 0.0}

    def _prepare_batch(self, batch: dict[str, Any], mode: Literal["train", "val", "test"]) -> dict[str, Any]:
        data = {}
        data["imgs"] = batch[self.modality].to(self.device, non_blocking=True).float() / 255
        data["targets"] = batch["mask"].to(self.device, dtype=torch.long)
        return data

    def _forward_batch(
        self, net: BaseNet, data: dict[str, Any], mode: Literal["train", "val", "test"]
    ) -> dict[str, Any]:
        imgs = data["imgs"]

        if mode in ("train", "val"):
            targets = data["targets"]
            preds = net(imgs)

            if self.geo_weight:
                loss, (ce_loss, geo_loss) = self.criterion(preds, targets)
                res = {"loss": loss, "ce_loss": ce_loss, "geo_loss": geo_loss, "preds": preds[0]}

                return res

            else:
                loss = self.criterion(preds, targets)
                mask_only = preds[0] if isinstance(preds, (tuple, list)) else preds
                return {"loss": loss, "preds": mask_only}

        else:
            preds = net(imgs)
            mask_only = preds[0] if isinstance(preds, (tuple, list)) else preds
            return {"preds": mask_only}

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
            if self.geo_weight:
                metrics["ce_loss"] = (metrics["ce_loss"] * batch_idx + res["ce_loss"].item()) / (batch_idx + 1)
                metrics["geo_loss"] = (metrics["geo_loss"] * batch_idx + res["geo_loss"].item()) / (batch_idx + 1)

            metrics["img_size"] = data["imgs"].size(2)

        elif mode == "val":
            loss = res["loss"]
            metrics["vloss"] = (metrics["vloss"] * batch_idx + loss.item()) / (batch_idx + 1)

            targets = data["targets"]
            preds = res["preds"].detach()
            preds = F.interpolate(preds, size=targets.shape[-2:], mode="bilinear", align_corners=False).max(dim=1)[1]

            self.metrics.update(targets.cpu().numpy(), preds.cpu().numpy())

            if batch_idx == self.val_batches - 1:  # calculate overall mIoU at the end of validation
                score = self.metrics.get_results()

                metrics["miou"] = score["Mean IoU"]
                metrics["mean acc"] = score["Mean Acc"]
                metrics["freqW acc"] = score["FreqW Acc"]
                metrics["overall acc"] = score["Overall Acc"]
                metrics["best_fitness"] = metrics["miou"]

                info["class iou"] = score["Class IoU"]

                self.metrics.reset()

        else:
            targets = data["targets"]
            preds = res["preds"].detach()
            preds = F.interpolate(preds, size=targets.shape[-2:], mode="bilinear", align_corners=False).max(dim=1)[1]

            self.metrics.update(targets.cpu().numpy(), preds.cpu().numpy())

            if batch_idx == self.test_batches - 1:  # calculate overall mIoU at the end of validation
                score = self.metrics.get_results()

                metrics["miou"] = score["Mean IoU"]
                metrics["mean acc"] = score["Mean Acc"]
                metrics["freqW acc"] = score["FreqW Acc"]
                metrics["overall acc"] = score["Overall Acc"]

                info["class iou"] = score["Class IoU"]

                self.metrics.reset()

    def write(self, batches: int, writer: SummaryWriter, res: dict[str, Any], metrics: dict[str, Any]) -> None:
        writer.add_scalar("loss/train_batch", metrics["tloss"], batches)  # batch loss
        if self.geo_weight:
            writer.add_scalar("ce_loss/train_batch", metrics["ce_loss"], batches)  # batch loss
            writer.add_scalar("geo_loss/train_batch", metrics["geo_loss"], batches)  # batch loss

    def close_dataloader_mosaic(self) -> None:
        if hasattr(self.train_loader.dataset, "close_mosaic"):
            LOGGER.info("Closing dataloader mosaic")
            self.train_loader.dataset.close_mosaic()

    def criterion(self, preds: torch.Tensor | tuple, targets: torch.Tensor) -> torch.Tensor:
        """Calculate the loss between predictions and targets using cross-entropy loss for per-pixel classification."""
        if self.geo_weight:
            masks, edges = preds[0], preds[1]

            masks = F.interpolate(masks, size=targets.shape[-2:], mode="bilinear", align_corners=False)
            edges = F.interpolate(edges, size=targets.shape[-2:], mode="bilinear", align_corners=False)
            ce_loss_raw = nn.CrossEntropyLoss(ignore_index=self.ignore_value)(masks, targets)

            gt_edges = generate_gt_edges(targets).unsqueeze(1).to(self.device)
            edge_loss_raw = nn.BCEWithLogitsLoss()(edges, gt_edges)
            total_loss = ce_loss_raw * self.ce_weight + edge_loss_raw * self.geo_weight

            return total_loss, (ce_loss_raw, edge_loss_raw)

        else:
            masks = preds[0] if isinstance(preds, (tuple, list)) else preds
            masks = F.interpolate(masks, size=targets.shape[-2:], mode="bilinear", align_corners=False)
            ce_loss = nn.CrossEntropyLoss(ignore_index=self.ignore_value)(masks, targets)

            return ce_loss * self.ce_weight

    @torch.no_grad()
    def predict(self, stream: str | VideoStream | WebcamStream, *args, **kwargs) -> None:
        """Make predictions from different special data stream."""
        super().predict(stream, *args, **kwargs)

        if isinstance(stream, (str, FilePath)):
            self._predict_single_frame(stream)

        elif isinstance(stream, (VideoStream, WebcamStream)):
            self._predict_stream(stream)

    def _predict_single_frame(self, img_path: FilePath) -> None:
        """Evaluate the single-frame image."""
        # read image
        img0 = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img0 is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)  # BGR -> RGB

        res_img = self._inference_and_preparation(img0)
        res_img_bgr = cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR)

        cv2.namedWindow("Segment result", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Segment result", 1280, 720)
        cv2.imshow("Segment result", res_img_bgr)
        cv2.waitKey(0)  # Wait for a key press
        cv2.destroyAllWindows()

    def _predict_stream(self, stream: VideoStream | WebcamStream) -> None:
        """Evaluate video images or live data streams."""
        # Calculate the exact inter-frame delay required for offline video
        # The camera is internally limited, so delay=1
        is_video = isinstance(stream, VideoStream)
        delay = max(1, int(1000 / stream.fps)) if is_video and stream.fps > 0 else 1

        LOGGER.info("Starting stream evaluation. Press 'q' to stop.")

        cv2.namedWindow("Segment result", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Segment result", 1280, 720)
        for frames in stream:
            if isinstance(frames, dict):
                frame = frames.get(self.modality)
                if frame is None:
                    raise ValueError(f"Stream does not contain required modality: {self.modality}")

            else:
                frame = frames

            f_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res_rgb = self._inference_and_preparation(f_rgb)
            show_frame = cv2.cvtColor(res_rgb, cv2.COLOR_RGB2BGR)

            cv2.imshow("Segment result", show_frame)

            if cv2.waitKey(delay) & 0xFF == ord("q"):
                LOGGER.info("Stream stopped by user.")
                break

        cv2.destroyAllWindows()

    def _inference_and_preparation(self, img0: np.ndarray) -> np.ndarray:
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

        mask_preds = preds[0] if isinstance(preds, (tuple, list)) else preds

        mask = torch.argmax(mask_preds, dim=1).squeeze(0)
        # rescale masks
        mask = rescale_masks(mask, (self.imgsz, self.imgsz), (h0, w0)).cpu().numpy()
        mask, _ = colour_mask(mask)

        return mask


# TODO
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


"""
Helper functions
"""


def dice_loss(pred, target, smooth=1e-5):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = 1 - ((2.0 * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth))
    return loss.mean()
