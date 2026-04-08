from typing import Literal, Mapping, Any

import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, ToTensor, Normalize
from ..yolo_v8 import YoloV8
from machine_learning.networks import BaseNet
from machine_learning.types.aliases import FilePath
from machine_learning.utils.logger import LOGGER
from machine_learning.utils.streams import VideoStream, WebcamStream
from machine_learning.utils.detection import (
    resize,
    non_max_suppression,
    pad_to_square,
    add_bboxes_to_image,
    rescale_boxes,
)


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
        super(YoloV8, self).train_epoch(epoch, writer, log_interval)

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
        super(YoloV8, self).validate()

        # log metrics
        metrics = {
            "class": "all",
            "images": 0,
            "vloss": 0.0,
            "best_fitness": 0.0,
            "labels": 0,
            "precision": 0.0,
            "recall": 0.0,
            "mAP.5": 0.0,
            "mAP.75": 0.0,
            "mAP.5-.95": 0.0,
        }
        info = {}
        stats = []
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
            metrics["vloss"] = (metrics["vloss"] * i + loss.item()) / (i + 1)
            batch_stats, img_num = self.get_statistics(preds, imgs.size(2), batch)
            stats.extend(batch_stats)
            metrics["images"] += img_num

            if i == self.val_batches - 1:
                # calculate mAP metrics
                mp, mr, map50, map75, map, nt = self.calculate_map(stats, info)
                metrics["mAP.5"] = map50
                metrics["mAP.75"] = map75
                metrics["mAP.5-.95"] = map
                metrics["precision"] = mp
                metrics["recall"] = mr
                metrics["labels"] = nt.sum()
                metrics["best_fitness"] = metrics["mAP.5-.95"]

            self.pbar_log("val", pbar, **metrics)

        return metrics, info

    @torch.no_grad()
    def eval(self) -> None:
        """Evaluate the preformece of the model on test dataset."""
        super(YoloV8, self).eval()

        if self.test_loader is None:
            raise ValueError("Test dataloader is not available.")

        metrics = {
            "class": "all",
            "images": 0,
            "labels": 0,
            "precision": 0.0,
            "recall": 0.0,
            "mAP.5": 0.0,
            "mAP.75": 0.0,
            "mAP.5-.95": 0.0,
        }
        info = {}
        stats = []
        self.print_metric_titles("val", metrics)

        pbar = tqdm(enumerate(self.test_loader), total=self.test_batches)
        for i, batch in pbar:
            imgs = batch["img"].to(self.device, non_blocking=True).float() / 255.0
            irs = batch["ir"].to(self.device, non_blocking=True).float() / 255.0  # convert ir to unit8 in advance

            preds = self.net(imgs, irs)

            # metrics
            batch_stats, img_num = self.get_statistics(preds, imgs.size(2), batch)
            stats.extend(batch_stats)
            metrics["images"] += img_num

            if i == self.test_batches - 1:
                # calculate mAP metrics
                mp, mr, map50, map75, map, nt = self.calculate_map(stats, info)
                metrics["mAP.5"] = map50
                metrics["mAP.75"] = map75
                metrics["mAP.5-.95"] = map
                metrics["precision"] = mp
                metrics["recall"] = mr
                metrics["labels"] = nt.sum()

            self.pbar_log("val", pbar, **metrics)

        return metrics, info

    @torch.no_grad()
    def predict(
        self,
        stream: dict[str, str] | VideoStream | WebcamStream,
        conf_thres: float | None = None,
        iou_thres: float | None = None,
        base: str = "img",
        *args,
        **kwargs,
    ):
        super(YoloV8, self).predict(stream)

        if isinstance(stream, dict):
            self._predict_single_frame(stream, conf_thres, iou_thres, base, *args, **kwargs)

        elif isinstance(stream, (VideoStream, WebcamStream)):
            self._predict_stream(stream, conf_thres, iou_thres, base, *args, **kwargs)

    def _predict_single_frame(
        self,
        paths: dict[str, str],
        conf_thres: float | None = None,
        iou_thres: float | None = None,
        base: str = "img",
        *args,
        **kwargs,
    ):
        """Evaluate the single-frame image."""
        # read image
        img_path = paths["img"]
        ir_path = paths["ir"]

        img0 = cv2.imread(paths["img"], cv2.IMREAD_COLOR)
        if img0 is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")

        ir0 = cv2.imread(paths["ir"], cv2.IMREAD_COLOR)
        if ir0 is None:
            raise FileNotFoundError(f"Failed to read image: {ir_path}")

        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        ir0 = cv2.cvtColor(ir0, cv2.COLOR_BGR2RGB)  # BGR -> RGB

        res = self._inference_and_preparation(img0, ir0, conf_thres, iou_thres, base, *args, **kwargs)

        res_bgr = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
        cv2.imshow("Prediction", res_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _predict_stream(
        self,
        stream: VideoStream | WebcamStream,
        conf_thres: float | None = None,
        iou_thres: float | None = None,
        base: str = "img",
        *args,
        **kwargs,
    ):
        """Evaluate video images or live data streams."""
        # Calculate the exact inter-frame delay required for offline video
        # The camera is internally limited, so delay=1
        is_video = isinstance(stream, VideoStream)
        delay = max(1, int(1000 / stream.fps)) if is_video and stream.fps > 0 else 1

        LOGGER.info("Starting stream evaluation. Press 'q' to stop.")

        for frames in stream:
            if not isinstance(frames, dict):
                raise ValueError("Multimodal detection requires a dictionary of frames from the stream.")

            img0 = cv2.cvtColor(frames["img"], cv2.COLOR_BGR2RGB)
            ir0 = cv2.cvtColor(frames["ir"], cv2.COLOR_BGR2RGB)
            res_rgb = self._inference_and_preparation(img0, ir0, conf_thres, iou_thres, base, *args, **kwargs)
            show_frame = cv2.cvtColor(res_rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow("Stream Evaluation", show_frame)

            if cv2.waitKey(delay) & 0xFF == ord("q"):
                LOGGER.info("Stream stopped by user.")
                break

        cv2.destroyAllWindows()

    def _inference_and_preparation(
        self,
        img0: np.ndarray,
        ir0: np.ndarray,
        conf_thres: float | None,
        iou_thres: float | None,
        base: str,
        *args,
        **kwargs,
    ) -> np.ndarray:
        """Core reasoning and rendering logic."""
        assert img0.shape[:2] == ir0.shape[:2], f"Input img and ir have different shapes: {img0.shape} vs {ir0.shape}."
        h0, w0, _ = img0.shape

        # scale to square
        padded_img = pad_to_square(img=img0, pad_values=(114, 114, 114))
        padded_ir = pad_to_square(img=ir0, pad_values=(0, 0, 0))

        # to tensor / normalize
        tfs = Compose([ToTensor(), Normalize(mean=[0, 0, 0], std=[1, 1, 1])])
        img = tfs(padded_img)
        img = resize(img, size=self.imgsz).unsqueeze(0).to(self.device)
        ir = tfs(padded_ir)
        ir = resize(ir, size=self.imgsz).unsqueeze(0).to(self.device)

        # input image to model
        preds = self.net(img, ir)

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

        # rescale to img coordinate
        detection = detections[0]

        if len(detection) > 0:
            bboxes, conf, cls = detection.split((4, 1, 1), dim=1)
            bboxes = rescale_boxes(np.array(bboxes.cpu(), dtype=np.float32), (self.imgsz, self.imgsz), (h0, w0))
            cls = [int(cid) for cid in cls]
        else:
            bboxes, conf, cls = np.zeros((0, 4)), [], []

        # visualization
        if base == "img":
            res_img = add_bboxes_to_image(img0, bboxes, cls, conf, self.class_names, *args, **kwargs)
        elif base == "ir":
            res_img = add_bboxes_to_image(ir0, bboxes, cls, conf, self.class_names, *args, **kwargs)

        return res_img
