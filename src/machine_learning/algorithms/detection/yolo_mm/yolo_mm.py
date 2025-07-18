from __future__ import annotations
from typing import Literal, Mapping, Any, Union

import cv2
import torch
from torch.utils.data import Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from machine_learning.networks import BaseNet
from machine_learning.algorithms.base import AlgorithmBase
from machine_learning.types.aliases import FilePath


class YoloMM(AlgorithmBase):
    def __init__(
        self,
        cfg: FilePath | Mapping[str, Any],
        models: Mapping[str, BaseNet],
        data: Mapping[str, Union[Dataset, Any]],
        name: str | None = "yolo_mm",
        device: Literal["cuda", "cpu", "auto"] = "auto",
    ) -> None:
        """
        Implementation of YoloMM object detection algorithm

        Args:
            cfg (str, dict): Configuration of the algorithm, it can be yaml file path or cfg dict.
            models (dict[str, BaseNet]): Models required by the YOLOMM algorithm, {"net": model}.
            data (Mapping[str, Union[Dataset, Any]]): Parsed specific dataset data, must including train dataset and val
            dataset, may contain data information of the specific dataset.
            name (str): Name of the algorithm. Defaults to "yolo_mm".
            device (Literal[&quot;cuda&quot;, &quot;cpu&quot;, &quot;auto&quot;], optional): Running device. Defaults to
            "auto"-automatic selection by algorithm.
        """
        super().__init__(cfg=cfg, models=models, data=data, name=name, device=device)

        # main parameters of the algorithm
        self.img_size = self.cfg["algorithm"].get("img_size", None)

        self.iou_threshold = self.cfg["algorithm"]["iou_threshold"]
        self.conf_threshold = self.cfg["algorithm"]["conf_threshold"]
        self.nms_threshold = self.cfg["algorithm"]["nms_threshold"]

        self.bw = self.cfg["algorithm"].get("box", 0.05)
        self.ow = self.cfg["algorithm"].get("cls", 1.0)
        self.cw = self.cfg["algorithm"].get("dfl", 0.5)

    def _configure_optimizers(self) -> None:
        opt_cfg = self._cfg["optimizer"]

        self.params = self.models["darknet"].parameters()

        if opt_cfg["type"] == "Adam":
            self._optimizers.update(
                {
                    "yolo": torch.optim.Adam(
                        params=self.params,
                        lr=opt_cfg["learning_rate"],
                        betas=(opt_cfg["beta1"], opt_cfg["beta2"]),
                        eps=opt_cfg["eps"],
                        weight_decay=opt_cfg["weight_decay"],
                    ),
                }
            )
        else:
            raise ValueError(f"Does not support optimizer:{opt_cfg['type']} currently.")

    def _configure_schedulers(self) -> None:
        sch_config = self._cfg["scheduler"]

        if sch_config.get("type") == "ReduceLROnPlateau":
            self._schedulers.update(
                {
                    "yolo": torch.optim.lr_scheduler.ReduceLROnPlateau(
                        self._optimizers["yolo"],
                        mode="min",
                        factor=sch_config.get("factor", 0.1),
                        patience=sch_config.get("patience", 10),
                    )
                }
            )

        if sch_config.get("type") == "LRWarnupDecay":

            def make_warmup_fn(warmup_epoch=15, decay_epochs=sch_config["epochs"], decay_scales=sch_config["scales"]):
                def fn(current_epoch):
                    if current_epoch < warmup_epoch:
                        return (current_epoch + 1) / warmup_epoch
                    else:
                        for i, epoch in enumerate(decay_epochs):
                            if current_epoch >= epoch:
                                return decay_scales[i]
                    return 1.0

                return fn

            self._schedulers.update(
                {"yolo": torch.optim.lr_scheduler.LambdaLR(self._optimizers["yolo"], lr_lambda=make_warmup_fn())}
            )

        else:
            print(f"Warning: Unknown scheduler type '{sch_config.get('type')}', no scheduler configured.")

    def train_epoch(self, epoch: int, writer: SummaryWriter, log_interval: int = 10) -> dict[str, float]:
        self.set_train()

        total_loss = 0.0
        scaler = GradScaler()

        for batch_idx, (imgs, thermals, iid_cls_bboxes) in enumerate(self.train_loader):
            imgs = imgs.to(self.device, non_blocking=True)
            thermals = thermals.to(self.device, non_blocking=True)
            targets = iid_cls_bboxes.to(self.device)  # (img_ids, class_ids, bboxes)

            self._optimizers["yolo"].zero_grad()

            with autocast(
                device_type=str(self.device)
            ):  # Ensure that the autocast scope correctly covers the forward computation
                fmap1, fmap2, fmap3 = self.models["nblitynet"](imgs)
                loss, loss_components = self.criterion(fmaps=[fmap1, fmap2, fmap3], targets=targets)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.params, self._cfg["optimizer"]["grad_clip"])
            scaler.step(self._optimizers["yolo"])
            scaler.update()

            total_loss += loss.item()

            batches = epoch * len(self.train_loader) + batch_idx
            if batch_idx % log_interval == 0:
                writer.add_scalar("iou loss/train_batch", loss_components[0].item(), batches)  # IOU loss
                writer.add_scalar("object loss/train_batch", loss_components[1].item(), batches)  # batch loss
                writer.add_scalar("class loss/train_batch", loss_components[2].item(), batches)  # batch loss
                writer.add_scalar("loss/train_batch", loss.item(), batches)  # batch loss

        avg_loss = total_loss / len(self.train_loader)

        return {"yolo loss": avg_loss}

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        self.set_eval()

        total_loss, total_iou_loss, total_obj_loss, total_cls_loss = 0.0, 0.0, 0.0, 0.0

        labels = []
        sample_metrics = []

        for imgs, iid_cls_bboxes in self.val_loader:
            imgs = imgs.to(self.device, non_blocking=True)
            targets = iid_cls_bboxes.to(self.device)  # (img_ids, class_ids, bboxes)
            labels += targets[:, 1].tolist()

            fmap1, fmap2, fmap3 = self.models["darknet"](imgs)

            # loss
            loss, loss_components = self.criterion(fmaps=[fmap1, fmap2, fmap3], targets=targets, img_size=imgs.size(2))
            total_loss += loss.item()
            total_iou_loss += loss_components[0].item()
            total_obj_loss += loss_components[1].item()
            total_cls_loss += loss_components[2].item()

            # metrics
            decodes = [
                self.fmap_decode(fmap, self.anchors[i], imgs.size(2)) for i, fmap in enumerate([fmap1, fmap2, fmap3])
            ]
            detections = non_max_suppression(
                decodes=decodes,
                conf_threshold=self.conf_threshold,
                nms_threshold=self.nms_threshold,
                device=self.device,
            )
            targets[:, 2:] = xywh2xyxy(targets[:, 2:])
            targets[:, 2:] *= imgs.size(2)

            sample_metrics += get_batch_statistics(
                detections, targets, iou_threshold=self.iou_threshold
            )  # [[true_positives, pred_scores, pred_clses], ...]

        avg_loss = total_loss / len(self.val_loader)
        avg_iou_loss = total_iou_loss / len(self.val_loader)
        avg_obj_loss = total_obj_loss / len(self.val_loader)
        avg_cls_loss = total_cls_loss / len(self.val_loader)

        if len(sample_metrics) == 0:  # No detections over whole validation set.
            print("---- No detections over whole validation set ----")
            return {
                "yolo loss": avg_loss,
                "save metric": avg_loss,
                "iou loss": avg_iou_loss,
                "object loss": avg_obj_loss,
                "class loss": avg_cls_loss,
                "mAP": 0.0,
            }

        # Concatenate sample statistics
        true_positives, pred_scores, pred_labels = [torch.cat(x, 0) for x in list(zip(*sample_metrics))]
        metrics_output = average_precision_per_cls(
            true_positives.cpu().numpy(), pred_scores.cpu().numpy(), pred_labels.cpu().numpy(), labels
        )

        result = {
            "yolo loss": avg_loss,
            "save metric": avg_loss,
            "iou loss": avg_iou_loss,
            "object loss": avg_obj_loss,
            "class loss": avg_cls_loss,
        }

        if metrics_output is not None:
            precision, recall, AP, f1, ap_class = metrics_output
            result.update(
                {
                    "precision": precision.mean(),
                    "recall": recall.mean(),
                    "f1": f1.mean(),
                    "mAP": AP.mean(),
                }
            )

        return result

    def eval(self) -> None:
        raise NotImplementedError(
            "Eval method is not implemented in the yolo-series algorithms, "
            + "please use detect method to detect objects in an image."
        )

    @torch.no_grad()
    def detect(self, img_path: FilePath, img_size: int, conf_threshold, nms_threshold) -> None:
        # read image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)
        origin_img_shape = img.shape

        # scale to square
        pad_img = pad_to_square(img=img, pad_values=0.1)

        # to tensor / normalize
        tfs = Compose([ToTensor(), Normalize(mean=[0.471, 0.448, 0.408], std=[0.234, 0.239, 0.242])])
        pad_img = tfs(pad_img)
        pad_img = resize(pad_img, size=img_size).unsqueeze(0).to(self.device)

        # input image to model
        fmap1, fmap2, fmap3 = self.models["darknet"](pad_img)

        # decode
        decodes = [
            self.fmap_decode(fmap, self.anchors[i], pad_img.size(2)) for i, fmap in enumerate([fmap1, fmap2, fmap3])
        ]
        detections = non_max_suppression(
            decodes=decodes,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            device=self.device,
        )

        # rescale to img coordiante
        detection = detections[0]
        detection[:, :4] = rescale_padded_boxes(detection[:, :4], pad_img.shape[2], origin_img_shape)

        bboxes = detection[:, :4]
        conf = detection[:, 4]
        cls = detection[:, 5].int()

        # visiualization
        visualize_img_with_bboxes(img, bboxes.cpu().numpy(), cls.cpu().numpy(), self.cfg["data"]["class_names"])

    def fmap_decode(self, fmap: torch.Tensor, anchors: torch.Tensor, img_size: int) -> torch.Tensor:
        """
        Decode the features output by the yolo detection net and map them to the image coordinate system (x, y, w, h).

        Args:
            fmap (torch.Tensor): the features output by the yolo detection net.
            img_size (int): the dim of image.
            anchors (torch.Tensor): the anchors corresponding to specific feature maps

        Returns:
            torch.Tensor: decode.
        """
        B, C, H, W = fmap.shape
        stride = img_size // H
        fmap = fmap.view(B, self.anchor_nums, -1, H, W).permute(0, 1, 3, 4, 2).contiguous()
        anchors = anchors.view(1, self.anchor_nums, 1, 1, 2)

        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=self.device),
            torch.arange(W, device=self.device),
            indexing="ij",
        )
        grid_xy = torch.stack((grid_x, grid_y), dim=-1).view(1, 1, H, W, 2).float()  # [H, W, 2]

        fmap[..., :2] = (torch.sigmoid(fmap[..., :2]) + grid_xy) * stride
        fmap[..., 2:4] = anchors * torch.exp(fmap[..., 2:4].clamp(-10, 10))
        fmap[..., 4:] = torch.sigmoid(fmap[..., 4:])

        return fmap.view(B, -1, self.cfg["data"]["class_nums"] + 5)

    @torch.no_grad()
    def prepare_target(self, targets: torch.Tensor, fh: int, fw: int, norm_anchors: torch.Tensor) -> tuple:
        """
        The target information of the original image is mapped to the fmap space to prepare for the loss calculation

        The reason for not mapping the feature space to the original image space is that the number of bounding boxes in
        the feature space is much larger than that in the target bounding box. For example, the number of bounding boxes
        in the 52*52 feature map is 52*52*3, and the computational complexity is high.

        Args:
            targets (torch.Tensor): the features output by the yolo detection net.
            fh (int): the height of fmap.
            fw (int): the weight of fmap.
            norm_anchors (torch.Tensor): the norm_anchors corresponding to specific fmap.

        Returns:
            torch.Tensor: decode.
        """
        num_bboxes = targets.shape[0]  # number of bboxes(targets)
        targets = targets.repeat(self.anchor_nums, 1, 1)  # targets [num_anchors, num_bboxes, 6]

        scale_tensor = torch.tensor([fw, fh, fw, fh], device=targets.device, dtype=torch.float32)
        targets[:, :, 2:6] *= scale_tensor
        anchor_ids = torch.arange(self.anchor_nums, device=self.device).repeat(num_bboxes, 1).T.view(-1, num_bboxes, 1)
        targets = torch.cat([anchor_ids, targets], dim=-1)  # (anchor_ids, img_id, class_id, x, y, w, h)

        # Filter the appropriate target box
        if num_bboxes:
            r = targets[:, :, 5:7] / norm_anchors[:, None]
            j = torch.max(r, 1.0 / r).max(2)[0] < self.anchor_scale_threshold
            targets = targets[j]
        else:
            targets = targets[0]

        anchor_ids, img_ids, tcls = targets[:, :3].long().T

        # The center point coordinates of the target bboxes in the feature map coordinate system
        gxy = targets[:, 3:5]
        gwh = targets[:, 5:7]
        gji = gxy.long()
        gj, gi = gji.T

        tbboxes = torch.cat((gxy - gji, gwh), 1)  # box
        indices = (img_ids, anchor_ids, gi.clamp_(0, fh - 1), gj.clamp_(0, fw - 1))
        norm_anchors = norm_anchors[anchor_ids]

        return tcls, tbboxes, indices, norm_anchors
