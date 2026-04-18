from typing import Any, Mapping, Literal

import cv2
import torch
import numpy as np

from machine_learning.utils.logger import LOGGER
from machine_learning.networks.base import BaseNet
from machine_learning.types.aliases import FilePath
from torchvision.transforms import Compose, ToTensor, Normalize
from machine_learning.utils.streams import VideoStream, WebcamStream
from machine_learning.utils.detect import pad_to_square, resize
from machine_learning.utils.segment import colour_mask, rescale_masks
from machine_learning.algorithms.segmentation import PerPixelSegmentation, MaskSegmentation


class MultimodalPerPixelSegmentation(PerPixelSegmentation):
    def __init__(
        self,
        cfg: FilePath | Mapping[str, Any],
        net: BaseNet | None = None,
        name: str | None = None,
        device: Literal["cuda", "cpu", "auto"] = "auto",
        amp: bool = True,
        ema: bool = True,
    ):
        """
        Implementation of Multimodal iamge segmentation algorithm.

        Args:
            cfg (FilePath, Mapping[str, Any]): Configuration of the algorithm, it can be yaml file path or cfg dict.
            net (BaseNet): Models required by the Multimodal algorithm.
            name (str): Name of the algorithm.
            device (Literal[&quot;cuda&quot;, &quot;cpu&quot;, &quot;auto&quot;], optional): Running device. Defaults to
            "auto"-automatic selection by algorithm.
            amp (bool): Whether to enable Automatic Mixed Precision. Defaults to False.
            ema (bool): Whether to enable Exponential Moving Average. Defaults to True.
        """
        super().__init__(cfg, net, name, device, amp, ema)

    def _prepare_batch(self, batch: dict[str, Any], mode: Literal["train", "val", "test"]) -> dict[str, Any]:
        """Prepare different batch data for different modes."""
        _ = mode

        data = {}
        data["imgs"] = batch["img"].to(self.device, non_blocking=True).float() / 255.0
        data["irs"] = batch["ir"].to(self.device, non_blocking=True).float() / 255.0  # convert ir to unit8 in advance
        data["targets"] = batch["mask"].to(self.device)

        return data

    def _forward_batch(
        self,
        net: BaseNet,
        data: dict[str, Any],
        mode: Literal["train", "val", "test"],
    ) -> dict[str, Any]:
        imgs = data["imgs"]
        irs = data["irs"]

        if mode in ("train", "val"):
            targets = data["targets"]
            # Loss calculation
            preds = net(imgs, irs)
            loss = self.criterion(preds, targets)

            return {"loss": loss, "preds": preds}
        else:
            preds = net(imgs, irs)

            return {"preds": preds}

    @torch.no_grad()
    def predict(self, stream: dict[str, str] | VideoStream | WebcamStream, *args, **kwargs) -> None:
        """Make predictions from different special data stream."""
        super().predict(stream, *args, **kwargs)

        if isinstance(stream, dict):
            self._predict_single_frame(stream)

    def _predict_single_frame(self, paths: dict[str, str]) -> None:
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

        res = self._inference_and_preparation(img0, ir0)

        res_bgr = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
        cv2.namedWindow("Segment result", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Segment result", 1280, 720)
        cv2.imshow("Segment result", res_bgr)
        cv2.waitKey(0)
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
            if not isinstance(frames, dict):
                raise ValueError("Multimodal detection requires a dictionary of frames from the stream.")

            img0 = cv2.cvtColor(frames["img"], cv2.COLOR_BGR2RGB)
            ir0 = cv2.cvtColor(frames["ir"], cv2.COLOR_BGR2RGB)
            res_rgb = self._inference_and_preparation(img0, ir0)
            show_frame = cv2.cvtColor(res_rgb, cv2.COLOR_RGB2BGR)

            cv2.imshow("Segment result", show_frame)

            if cv2.waitKey(delay) & 0xFF == ord("q"):
                LOGGER.info("Stream stopped by user.")
                break

        cv2.destroyAllWindows()

    def _inference_and_preparation(self, img0: np.ndarray, ir0: np.ndarray) -> np.ndarray:
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

        mask = torch.argmax(preds, dim=1).squeeze(0)
        # rescale masks
        mask = rescale_masks(mask, (self.imgsz, self.imgsz), (h0, w0)).cpu().numpy()
        mask, _ = colour_mask(mask)

        return mask


class MultimodalMaskSegmentation(MaskSegmentation):
    def __init__(
        self,
        cfg,
        net=None,
        name=None,
        device="auto",
        amp=True,
        ema=True,
        modality="img",
    ):
        super().__init__(cfg, net, name, device, amp, ema, modality)
        self.modality = modality
        self.loss_weight = self.cfg["algorithm"].get("loss_weight", 1.0)
        self.ignore_value = self.cfg["data"].get("ignore_value", -100)
