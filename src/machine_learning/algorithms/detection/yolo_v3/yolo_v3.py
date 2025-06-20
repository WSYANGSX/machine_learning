from __future__ import annotations
from typing import Literal, Iterable, Mapping, Any
from itertools import chain

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from machine_learning.models import BaseNet
from machine_learning.algorithms.base import AlgorithmBase
from machine_learning.types.aliases import FilePath
from machine_learning.utils.detection import bbox_iou


class YoloV3(AlgorithmBase):
    def __init__(
        self,
        cfg: FilePath | Mapping[str, Any],
        models: Mapping[str, BaseNet],
        name: str | None = "yolo_v3",
        device: Literal["cuda", "cpu", "auto"] = "auto",
    ) -> None:
        """
        Implementation of YoloV3 object detection algorithm

        Args:
            cfg (str, dict): Configuration of the algorithm, it can be yaml file path or cfg dict.
            models (dict[str, BaseNet]): Models required by the YOLOv3 algorithm, {"darknet": model1, "fpn": model2}.
            name (str): Name of the algorithm. Defaults to "yolo_v3".
            device (Literal[&quot;cuda&quot;, &quot;cpu&quot;, &quot;auto&quot;], optional): Running device. Defaults to
            "auto"-automatic selection by algorithm.
        """
        super().__init__(cfg=cfg, models=models, name=name, device=device)

        # main parameters of the algorithm
        self.anchor_nums = self.cfg["algorithm"]["anchor_nums"]
        self.anchor_sizes = self.cfg["algorithm"]["anchor_sizes"]
        self.default_img_size = self.cfg["algorithm"]["default_img_size"]
        self.iou_threshold = self.cfg["algorithm"]["iou_threshold"]
        self.obj_exist_threshold = self.cfg["algorithm"]["obj_exist_threshold"]
        self.b_weiget = self.cfg["algorithm"].get("b_weiget", 0.05)
        self.o_weiget = self.cfg["algorithm"].get("o_weiget", 1.0)
        self.c_weiget = self.cfg["algorithm"].get("c_weiget", 0.5)

        # parameters that varys due to difference of dataset
        self.class_names = None

    def _configure_optimizers(self) -> None:
        opt_cfg = self._cfg["optimizer"]

        self.params = chain(self.models["darknet"].parameters(), self.models["fpn"].parameters())

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
            ValueError(f"Does not support optimizer:{opt_cfg['type']} currently.")

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

    def train_epoch(self, epoch: int, writer: SummaryWriter, log_interval: int = 10) -> dict[str, float]:
        self.set_train()

        total_loss = 0.0

        for batch_idx, (img, bboxes, category_ids, indices) in enumerate(self.train_loader):
            if check_data_integrity(img, bboxes, category_ids, self.class_nums):
                print(f"Epoch: {epoch}, Batch: {batch_idx}, invalid data detected, skipping.")
                continue

            img = img.to(self.device, non_blocking=True)
            cls_iids_bboxes = torch.cat([category_ids.view(-1, 1), indices.view(-1, 1), bboxes], dim=-1).to(
                self.device
            )  # (class_ids, img_ids, bboxes)

            self._optimizers["yolo"].zero_grad()

            skips = self.models["darknet"](img)
            fimgs = self.models["fpn"](*skips)

            det_ls, norm_anchors_ls = self.feature_decode(fimgs, img_size=img.shape[2])

            loss = self.criterion(det_ls, norm_anchors_ls, cls_iids_bboxes)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.params, self._cfg["optimizer"]["grad_clip"])
            self._optimizers["yolo"].step()

            total_loss += loss.item()

            if batch_idx % log_interval == 0:
                writer.add_scalar(
                    "loss/train_batch", loss.item(), epoch * len(self.train_loader) + batch_idx
                )  # batch loss

        avg_loss = total_loss / len(self.train_loader)

        return {"yolo": avg_loss}

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        self.set_eval()

        total_loss = 0.0

        for img, bboxes, category_ids, indices in self.train_loader:
            img = img.to(self.device, non_blocking=True)
            cls_iids_bboxes = torch.cat([category_ids.view(-1, 1), indices.view(-1, 1), bboxes], dim=-1).to(
                self.device
            )  # (class_id, img_id, bboxes)

            skips = self.models["darknet"](img)
            fimgs = self.models["fpn"](*skips)

            det_ls, norm_anchors_ls = self.feature_decode(fimgs, img_size=img.shape[2])

            total_loss += self.criterion(det_ls, norm_anchors_ls, cls_iids_bboxes).item()

        avg_loss = total_loss / len(self.val_loader)

        return {"yolo": avg_loss, "save": avg_loss}

    @torch.no_grad()
    def eval(self, img_path: FilePath) -> None:
        pass

    def feature_decode(self, features: Iterable[torch.Tensor], img_size: int) -> tuple[list]:
        detection_ls, norm_anchors_ls = [], []

        for feature in features:
            # [B,C,H,W] -> [B,H,W,C]
            detection = feature.permute(0, 2, 3, 1).contiguous()
            B, H, W, _ = detection.shape
            stride = img_size // H  # compute stride

            # anchors choose
            if H == 52:
                norm_anchors = torch.tensor(self.anchor_sizes[:3], dtype=torch.float32, device=self.device) / stride
            elif H == 26:
                norm_anchors = torch.tensor(self.anchor_sizes[3:6], dtype=torch.float32, device=self.device) / stride
            else:
                norm_anchors = torch.tensor(self.anchor_sizes[6:9], dtype=torch.float32, device=self.device) / stride

            # Construct the offset matrix, paying attention to the "ij" form and the "xy" form
            grid_x, grid_y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
            grid_xy = torch.stack((grid_x, grid_y), dim=-1).float().to(self.device)  # [H, W, 2]

            # [H, W, 1, 2] -> [B, H, W, num_anchors, 2]
            grid_xy = grid_xy.view(1, H, W, 1, 2).expand(B, H, W, self.anchor_nums, 2)

            # [num_anchors, 2] -> [1, 1, 1, num_anchors, 2]
            norm_wh = norm_anchors.view(1, 1, 1, self.anchor_nums, 2)

            # [B, H, W, num_anchors, (5 + num_classes)]
            detection = detection.view(B, H, W, self.anchor_nums, -1)

            # Decompose the original detection tensor. In-place operations on the tensor will disrupt the
            # gradient and lead to calculation errors
            xy = detection[..., :2]  # Center point offset
            wh = detection[..., 2:4].clamp(-10, 10)  # Width and height offset
            obj = detection[..., 4:5]  # Target Confidence Level
            cls = detection[..., 5:]  # Classification Probability

            new_xy = (
                torch.sigmoid(xy) + grid_xy
            )  # The coordinates of center point on the coordinate system of feature map
            new_wh = norm_wh * torch.exp(wh)  # The width and height of bboxes on the feature map coordinate system
            new_obj = torch.sigmoid(obj)  # Map the existence of obj to 0-1
            new_cls = torch.sigmoid(cls)  # Whether the cls is correctly mapped to 0-1

            # reshape tensor -> [B, A, H, W, (C/A)]
            detection_decoded = torch.cat([new_xy, new_wh, new_obj, new_cls], dim=-1)
            detection_decoded = detection_decoded.permute(0, 3, 1, 2, 4).contiguous()  # [B, num_anchors, H, W, ...]

            detection_ls.append(detection_decoded)
            norm_anchors_ls.append(norm_anchors)

        return detection_ls, norm_anchors_ls

    def prepare_targets(
        self,
        dets_ls: list[torch.Tensor],
        norm_anchors_ls: list[torch.Tensor],
        cls_iids_bboxes: torch.Tensor,
    ) -> tuple:
        """
        The target information of the original image is mapped to the feature space to prepare for the loss calculation

        The reason for not mapping the feature space to the original image space is that the number of bounding boxes in
        the feature space is much larger than that in the target bounding box. For example, the number of bounding boxes
        in the 52*52 feature map is 52*52*3, and the computational complexity is high.
        """
        tcls, tbboxes, indices, tanchors = [], [], [], []

        num_bboxes = cls_iids_bboxes.shape[0]  # number of bboxes(targets)

        for _, (dets, norm_anchors) in enumerate(zip(dets_ls, norm_anchors_ls)):
            det_height, det_width = dets.shape[2], dets.shape[3]  # det [B, A, H, W, (C / A)]

            targets = cls_iids_bboxes.repeat(self.anchor_nums, 1, 1)  # targets [num_anchors, num_bboxes, 6]
            targets[:, :, 2:6] *= cls_iids_bboxes.new([det_width, det_height]).repeat(num_bboxes, 2)

            anchor_ids = (
                torch.arange(self.anchor_nums, device=self.device)
                .view(self.anchor_nums, 1)
                .repeat(1, num_bboxes)
                .view(self.anchor_nums, num_bboxes, 1)
            )

            targets = torch.cat(
                [anchor_ids, targets], dim=-1
            )  # targets [num_anchors, num_bboxes, 7] (anchor_ids, class_id, img_id, x, y, w, h)

            # Filter the appropriate target box
            if num_bboxes:
                r = targets[:, :, 5:7] / norm_anchors[:, None]
                j = torch.max(r, 1.0 / r).max(2)[0] < 4
                targets = targets[j]

            anchor_ids, cls_ids, img_ids = targets[:, :3].long().T

            # The center point coordinates of the target bboxes in the feature map coordinate system
            gxy = targets[:, 3:5]
            # The width and height of the target bboxes in the coordinate system of the feature map
            gwh = targets[:, 5:7]
            gij = gxy.long()
            gi, gj = gij.T

            tbboxes.append(torch.cat((gxy, gwh), 1))  # box
            tcls.append(cls_ids)
            tanchors.append(norm_anchors[anchor_ids])
            indices.append((img_ids, anchor_ids, gj.clamp_(0, det_height - 1), gi.clamp_(0, det_width - 1)))

        return tcls, tbboxes, indices, tanchors

    def criterion(
        self,
        dets_ls: list[torch.Tensor],
        anchors_ls: list[torch.Tensor],
        cls_iids_bboxes: torch.Tensor,
    ) -> torch.Tensor:
        tcls, tbboxes, indices, _ = self.prepare_targets(dets_ls, anchors_ls, cls_iids_bboxes)

        cls_loss = torch.scalar_tensor(0, dtype=torch.float32, device=self.device)
        bbox_loss = torch.scalar_tensor(0, dtype=torch.float32, device=self.device)
        obj_loss = torch.scalar_tensor(0, dtype=torch.float32, device=self.device)

        # 使用BCELoss
        BCEcls = nn.BCELoss()
        BCEobj = nn.BCELoss()

        for i, det in enumerate(dets_ls):
            img_ids, anchor_ids, grid_j, grid_i = indices[i]
            num_bboxes = img_ids.shape[0]

            # Use detach() to create tobj to avoid gradient propagation issues.
            tobj = torch.zeros_like(det[..., 0], device=self.device)

            if num_bboxes:
                ps = det[img_ids, anchor_ids, grid_j, grid_i]  # det [B, A, H, W, (C / A)]
                pxy, pwh = ps[:, :2], ps[:, 2:4]
                pbox = torch.cat((pxy, pwh), 1)
                iou = bbox_iou(pbox, tbboxes[i], bbox_format="coco", iou_type="ciou")

                if torch.isnan(iou).any() or torch.isinf(iou).any():
                    print("iou has nan value.")

                bbox_loss += (1.0 - iou).mean()  # iou loss

                tobj[img_ids, anchor_ids, grid_j, grid_i] = iou.detach().clamp(0.0, 1.0).type(tobj.dtype)
                if ps.size(1) > 5:
                    targets = torch.zeros_like(ps[:, 5:], device=self.device)
                    targets[range(num_bboxes), tcls[i]] = 1.0

                    class_preds = ps[:, 5:]
                    cls_loss += BCEcls(class_preds, targets)

            obj_preds = det[..., 4]
            obj_loss += BCEobj(obj_preds, tobj.detach())

        loss = self.b_weiget * bbox_loss + self.o_weiget * obj_loss + self.c_weiget * cls_loss

        return loss


"""
Helper function
"""


def check_data_integrity(img: torch.Tensor, bboxes: torch.Tensor, category_ids: torch.Tensor, class_nums: int) -> bool:
    "Check whether the input data is valid"
    # check img data
    if torch.isnan(img).any() or torch.isinf(img).any():
        return True

    # check bboxes data
    if (bboxes[..., 2:] <= 0).any():  # height and width less than zero
        return True
    if (bboxes[..., :2] < 0).any() or (bboxes[..., :2] > 1).any():  # value out of range [0,1]
        return True

    # check category_ids
    if (category_ids < 0).any() or (category_ids >= class_nums).any():
        return True

    return False
