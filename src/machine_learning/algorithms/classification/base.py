from typing import Any, Mapping, Literal

import cv2
import torch
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, ToTensor, Normalize

from machine_learning.utils.logger import LOGGER
from machine_learning.networks.base import BaseNet
from machine_learning.types.aliases import FilePath
from machine_learning.algorithms.base import AlgorithmBase
from machine_learning.utils.classify import ClassificationMetrics
from machine_learning.utils.streams import VideoStream, WebcamStream
from machine_learning.utils.detect import pad_to_square, resize


class Classification(AlgorithmBase):
    """
    Base class for classification algorithms.
    """

    def __init__(
        self,
        cfg: FilePath | Mapping[str, Any],
        name: str | None = None,
        device: Literal["cuda", "cpu", "auto"] = "auto",
        amp: bool = True,
        ema: bool = True,
    ):
        super().__init__(cfg=cfg, name=name, device=device, amp=amp, ema=ema)

        self.modality = self.cfg["algorithm"].get("modality", "img")

        # loss weight
        self.loss_weight = self.cfg["algorithm"].get("loss_weight")
        self.imgsz = self.cfg["algorithm"]["imgsz"]
        self.class_names = self.dataset_cfg["class_names"]
        self.nc = self.dataset_cfg["nc"]

        self.metrics = ClassificationMetrics(nc=self.nc)

    def _init_metrics(self, mode: Literal["train", "val", "test"]) -> dict[str, Any]:
        if mode == "train":
            return {"tloss": 0.0, "img_size": None}

        elif mode == "val":
            return {
                "vloss": 0.0,
                "best_fitness": 0.0,
                "accuracy": 0.0,
                "macro Precision": 0.0,
                "macro Recall": 0.0,
                "macro F1": 0.0,
                "micro Precision": 0.0,
                "micro Recall": 0.0,
                "micro F1": 0.0,
                "weighted Precision": 0.0,
                "weighted Recall": 0.0,
                "weighted F1": 0.0,
            }
        else:
            return {
                "accuracy": 0.0,
                "macro Precision": 0.0,
                "macro Recall": 0.0,
                "macro F1": 0.0,
                "micro Precision": 0.0,
                "micro Recall": 0.0,
                "micro F1": 0.0,
                "weighted Precision": 0.0,
                "weighted Recall": 0.0,
                "weighted F1": 0.0,
            }

    def _prepare_batch(self, batch: dict[str, Any], mode: Literal["train", "val", "test"]) -> dict[str, Any]:
        data = {}
        data["imgs"] = batch[self.modality].to(self.device, non_blocking=True).float() / 255
        data["targets"] = batch["cls"].to(self.device, dtype=torch.long)
        return data

    def _forward_batch(
        self, nets: dict[str, BaseNet], data: dict[str, Any], mode: Literal["train", "val", "test"]
    ) -> dict[str, Any]:
        imgs = data["imgs"]
        net = nets["net"]

        if mode in ("train", "val"):
            targets = data["targets"]
            preds = net(imgs)

            loss = self.criterion(preds, targets)
            return {"loss": loss, "preds": preds}

        else:
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
            metrics["img_size"] = data["imgs"].size(2)

        elif mode == "val":
            loss = res["loss"]
            metrics["vloss"] = (metrics["vloss"] * batch_idx + loss.item()) / (batch_idx + 1)

            targets = data["targets"]
            preds = torch.argmax(res["preds"].detach(), dim=1)
            self.metrics.update(targets.cpu().numpy(), preds.cpu().numpy())

            if batch_idx == self.val_batches - 1:  # calculate overall mIoU at the end of validation
                score = self.metrics.get_results()

                metrics["accuracy"] = score["Accuracy"]
                metrics["macro Precision"] = score["Macro Precision"]
                metrics["macro Recall"] = score["Macro Recall"]
                metrics["macro F1"] = score["Macro F1"]
                metrics["micro Precision"] = score["Micro Precision"]
                metrics["micro Recall"] = score["Micro Recall"]
                metrics["micro F1"] = score["Micro F1"]
                metrics["weighted Precision"] = score["Weighted Precision"]
                metrics["weighted Recall"] = score["Weighted Recall"]
                metrics["weighted F1"] = score["Weighted F1"]
                metrics["best_fitness"] = metrics["accuracy"]

                info["class Precision"] = score["Class Precision"]
                info["class Recall"] = score["Class Recall"]
                info["class F1"] = score["Class F1"]

                self.metrics.reset()

        else:
            targets = data["targets"]
            preds = torch.argmax(res["preds"].detach(), dim=1)
            self.metrics.update(targets.cpu().numpy(), preds.cpu().numpy())

            if batch_idx == self.test_batches - 1:  # calculate overall mIoU at the end of validation
                score = self.metrics.get_results()

                metrics["accuracy"] = score["Accuracy"]
                metrics["macro Precision"] = score["Macro Precision"]
                metrics["macro Recall"] = score["Macro Recall"]
                metrics["macro F1"] = score["Macro F1"]
                metrics["micro Precision"] = score["Micro Precision"]
                metrics["micro Recall"] = score["Micro Recall"]
                metrics["micro F1"] = score["Micro F1"]
                metrics["weighted Precision"] = score["Weighted Precision"]
                metrics["weighted Recall"] = score["Weighted Recall"]
                metrics["weighted F1"] = score["Weighted F1"]

                info["class Precision"] = score["Class Precision"]
                info["class Recall"] = score["Class Recall"]
                info["class F1"] = score["Class F1"]

                self.metrics.reset()

    def write(self, batches: int, writer: SummaryWriter, res: dict[str, Any], metrics: dict[str, Any]) -> None:
        writer.add_scalar("loss/train_batch", metrics["tloss"], batches)  # batch loss

    def criterion(self, preds: torch.Tensor | tuple, targets: torch.Tensor) -> torch.Tensor:
        """Calculate the loss between predictions and targets using cross-entropy loss for per-pixel classification."""
        loss = nn.CrossEntropyLoss()(preds, targets)

        return loss * self.loss_weight

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
        img0 = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img0 is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")

        img_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)

        pred_cls_id, conf = self._inference_and_preparation(img_rgb)
        label_name = self.class_names[pred_cls_id]

        text = f"{label_name}: {conf:.2f}"
        cv2.putText(img0, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        cv2.namedWindow("Classification result", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Classification result", 1280, 720)
        cv2.imshow("Classification result", img0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _predict_stream(self, stream: VideoStream | WebcamStream) -> None:
        """Evaluate video images or live data streams."""
        # Calculate the exact inter-frame delay required for offline video
        # The camera is internally limited, so delay=1
        is_video = isinstance(stream, VideoStream)
        delay = max(1, int(1000 / stream.fps)) if is_video and stream.fps > 0 else 1

        LOGGER.info("Starting stream evaluation. Press 'q' to stop.")

        cv2.namedWindow("Classification result", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Classification result", 1280, 720)
        for frames in stream:
            if isinstance(frames, dict):
                frame = frames.get(self.modality)
                if frame is None:
                    raise ValueError(f"Stream does not contain required modality: {self.modality}")

            else:
                frame = frames

            f_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pred_cls_id, conf = self._inference_and_preparation(f_rgb)
            label_name = self.class_names[pred_cls_id]
            text = f"{label_name}: {conf:.2f}"
            cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            show_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            cv2.imshow("Classification result", show_frame)

            if cv2.waitKey(delay) & 0xFF == ord("q"):
                LOGGER.info("Stream stopped by user.")
                break

        cv2.destroyAllWindows()

    def _inference_and_preparation(self, img0: np.ndarray) -> tuple[int, float]:
        """Core reasoning and returning class index and confidence."""
        padded_img = pad_to_square(img=img0, pad_values=(114, 114, 114))

        tfs = Compose([ToTensor(), Normalize(mean=[0, 0, 0], std=[1, 1, 1])])
        img = tfs(padded_img)
        img = resize(img, size=self.imgsz).unsqueeze(0).to(self.device)

        preds = self.net(img)  # [1, NC]

        # Convert Logits to probabilities and obtain the maximum probability and its index
        probs = torch.softmax(preds, dim=1)
        conf, pred_cls = torch.max(probs, dim=1)

        return pred_cls.item(), conf.item()
