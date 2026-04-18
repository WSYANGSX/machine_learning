from typing import Literal, Mapping, Any

import cv2
import torch
import numpy as np

from torchvision.transforms import Compose, ToTensor, Normalize
from ..yolo_v8 import YoloV8
from machine_learning.networks import BaseNet
from machine_learning.types.aliases import FilePath
from machine_learning.utils.logger import LOGGER
from machine_learning.utils.streams import VideoStream, WebcamStream
from machine_learning.utils.detect import (
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

    def _prepare_batch(self, batch: dict[str, Any], mode: Literal["train", "val", "test"]) -> dict[str, Any]:
        """Prepare different batch data for different modes."""
        data = {}

        imgs = batch["img"].to(self.device, non_blocking=True).float() / 255.0
        irs = batch["ir"].to(self.device, non_blocking=True).float() / 255.0  # convert ir to unit8 in advance
        data["imgs"] = imgs
        data["irs"] = irs

        if mode in ("train", "val"):
            targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1).to(
                self.device
            )  # (img_ids, class_ids, bboxes)
            data["targets"] = targets

        return data

    def _forward_batch(
        self,
        net: BaseNet,
        data: dict[str, Any],
        mode: Literal["train", "val", "test"],
    ) -> dict[str, Any]:
        if mode in ("train", "val"):
            imgs = data["imgs"]
            irs = data["irs"]
            targets = data["targets"]
            imgs_shape = imgs.shape

            # Loss calculation
            preds = net(imgs, irs)
            loss, lc = self.criterion(preds=preds, targets=targets, imgs_shape=imgs_shape)

            return {"loss": loss, "lc": lc, "preds": preds}
        else:
            imgs = data["imgs"]
            irs = data["irs"]
            preds = net(imgs, irs)

            return {"preds": preds}

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
        super().predict(stream, *args, **kwargs)

        if isinstance(stream, dict):
            self._predict_single_frame(stream, conf_thres, iou_thres, base)

    def _predict_single_frame(
        self,
        paths: dict[str, str],
        conf_thres: float | None = None,
        iou_thres: float | None = None,
        base: str = "img",
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

        res = self._inference_and_preparation(img0, ir0, conf_thres, iou_thres, base)

        res_bgr = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
        cv2.namedWindow("Prediction", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Prediction", 1280, 720)
        cv2.imshow("Prediction", res_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _predict_stream(
        self,
        stream: VideoStream | WebcamStream,
        conf_thres: float | None = None,
        iou_thres: float | None = None,
        base: str = "img",
    ):
        """Evaluate video images or live data streams."""
        # Calculate the exact inter-frame delay required for offline video
        # The camera is internally limited, so delay=1
        is_video = isinstance(stream, VideoStream)
        delay = max(1, int(1000 / stream.fps)) if is_video and stream.fps > 0 else 1

        LOGGER.info("Starting stream evaluation. Press 'q' to stop.")

        cv2.namedWindow("Prediction", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Prediction", 1280, 720)

        for frames in stream:
            if not isinstance(frames, dict):
                raise ValueError("Multimodal detection requires a dictionary of frames from the stream.")

            img0 = cv2.cvtColor(frames["img"], cv2.COLOR_BGR2RGB)
            ir0 = cv2.cvtColor(frames["ir"], cv2.COLOR_BGR2RGB)
            res_rgb = self._inference_and_preparation(img0, ir0, conf_thres, iou_thres, base)
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
            res_img = add_bboxes_to_image(img0, bboxes, cls, conf, self.class_names)
        elif base == "ir":
            res_img = add_bboxes_to_image(ir0, bboxes, cls, conf, self.class_names)

        return res_img
