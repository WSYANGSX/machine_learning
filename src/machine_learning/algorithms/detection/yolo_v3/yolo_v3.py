from __future__ import annotations
from typing import Literal, Mapping, Any
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
        self.anchor_scale_threshold = self.cfg["algorithm"]["anchor_scale_threshold"]

        self.b_weiget = self.cfg["algorithm"].get("b_weiget", 0.05)
        self.o_weiget = self.cfg["algorithm"].get("o_weiget", 1.0)
        self.c_weiget = self.cfg["algorithm"].get("c_weiget", 0.5)

        # parameters that varys due to difference of dataset
        # if the algorithm needs info from dataset, you need to return them from data parser
        self.class_names = None
        self.class_nums = None

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

        for batch_idx, (img, iid_cls_bboxes) in enumerate(self.train_loader):
            if check_data_integrity(img, iid_cls_bboxes[:, 2:6], iid_cls_bboxes[:, 1], self.class_nums):
                print(f"Epoch: {epoch}, Batch: {batch_idx}, invalid data detected, skipping.")
                continue

            img = img.to(self.device, non_blocking=True)
            iid_cls_bboxes = iid_cls_bboxes.to(self.device)  # (img_ids, class_ids, bboxes)

            self._optimizers["yolo"].zero_grad()

            skips = self.models["darknet"](img)
            fmap1, fmap2, fmap3 = self.models["fpn"](*skips)  # 52x52, 26x26, 13x13

            decode1, norm_anchors1 = self.fmap_decode(fmap1, img_size=img.shape[2])
            decode2, norm_anchors2 = self.fmap_decode(fmap2, img_size=img.shape[2])
            decode3, norm_anchors3 = self.fmap_decode(fmap3, img_size=img.shape[2])

            loss, loss_components = self.criterion(
                decode_ls=[decode1, decode2, decode3],
                norm_anchors_ls=[norm_anchors1, norm_anchors2, norm_anchors3],
                iid_cls_bboxes=iid_cls_bboxes,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.params, self._cfg["optimizer"]["grad_clip"])
            self._optimizers["yolo"].step()

            total_loss += loss.item()

            if batch_idx % log_interval == 0:
                writer.add_scalar(
                    "iou loss/train_batch", loss_components[0].item(), epoch * len(self.train_loader) + batch_idx
                )  # IOU loss
                writer.add_scalar(
                    "object loss/train_batch", loss_components[1].item(), epoch * len(self.train_loader) + batch_idx
                )  # batch loss
                writer.add_scalar(
                    "class loss/train_batch", loss_components[2].item(), epoch * len(self.train_loader) + batch_idx
                )  # batch loss
                writer.add_scalar(
                    "loss/train_batch", loss.item(), epoch * len(self.train_loader) + batch_idx
                )  # batch loss

        avg_loss = total_loss / len(self.train_loader)

        return {"yolo": avg_loss}

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        self.set_eval()

        total_loss = 0.0
        total_iou_loss = 0.0
        total_obj_loss = 0.0
        total_cls_loss = 0.0

        for img, iid_cls_bboxes in self.train_loader:
            img = img.to(self.device, non_blocking=True)
            iid_cls_bboxes = iid_cls_bboxes.to(self.device)  # (img_ids, class_ids, bboxes)

            skips = self.models["darknet"](img)
            fmap1, fmap2, fmap3 = self.models["fpn"](*skips)  # 52x52, 26x26, 13x13

            decode1, norm_anchors1 = self.fmap_decode(fmap1, img_size=img.shape[2])
            decode2, norm_anchors2 = self.fmap_decode(fmap2, img_size=img.shape[2])
            decode3, norm_anchors3 = self.fmap_decode(fmap3, img_size=img.shape[2])

            loss, loss_components = self.criterion(
                decode_ls=[decode1, decode2, decode3],
                norm_anchors_ls=[norm_anchors1, norm_anchors2, norm_anchors3],
                iid_cls_bboxes=iid_cls_bboxes,
            )

            total_loss += loss.item()
            total_iou_loss += loss_components[0].item()
            total_obj_loss += loss_components[1].item()
            total_cls_loss += loss_components[2].item()

        avg_loss = total_loss / len(self.val_loader)
        avg_iou_loss = total_iou_loss / len(self.val_loader)
        avg_obj_loss = total_obj_loss / len(self.val_loader)
        avg_cls_loss = total_cls_loss / len(self.val_loader)

        return {
            "yolo": avg_loss,
            "save": avg_loss,
            "iou loss": avg_iou_loss,
            "object loss": avg_obj_loss,
            "class loss": avg_cls_loss,
        }

    @torch.no_grad()
    def eval(self, img_path: FilePath) -> None:
        self.set_eval()

    def fmap_decode(self, feature_map: torch.Tensor, img_dim: int) -> tuple[list]:
        # [B,C,H,W] -> [B,H,W,C]
        fmap = feature_map.permute(0, 2, 3, 1).contiguous()
        B, H, W, _ = fmap.shape
        stride = img_dim // H  # compute stride

        # anchors choose, normalize to feature map coordinate
        if H == 52:
            norm_anchors = torch.tensor(self.anchor_sizes[:3], dtype=torch.float32, device=self.device) / stride
        elif H == 26:
            norm_anchors = torch.tensor(self.anchor_sizes[3:6], dtype=torch.float32, device=self.device) / stride
        else:
            norm_anchors = torch.tensor(self.anchor_sizes[6:9], dtype=torch.float32, device=self.device) / stride

        # Construct the offset matrix, paying attention to the "ij" form and the "xy" form
        grid_x, grid_y = torch.meshgrid(
            torch.arange(W, device=self.device), torch.arange(H, device=self.device), indexing="xy"
        )
        grid_xy = torch.stack((grid_x, grid_y), dim=-1)  # [H, W, 2]

        # [H, W, 1, 2] -> [B, H, W, num_anchors, 2]
        grid_xy = grid_xy.view(1, H, W, 1, 2).expand(B, H, W, self.anchor_nums, 2)

        # [num_anchors, 2] -> [1, 1, 1, num_anchors, 2]
        norm_wh = norm_anchors.view(1, 1, 1, self.anchor_nums, 2)

        # [B, H, W, num_anchors, (5 + num_classes)]
        fmap = fmap.view(B, H, W, self.anchor_nums, -1)

        # Decompose the original fmap tensor. In-place operations on the tensor will disrupt the
        # gradient and lead to calculation errors
        xy = fmap[..., :2]  # Center point offset
        # wh = fmap[..., 2:4].clamp(-10, 10)  # Width and height offset
        wh = fmap[..., 2:4]
        obj = fmap[..., [4]]  # Target Confidence Level
        cls = fmap[..., 5:]  # Classification Probability

        new_xy = torch.sigmoid(xy) + grid_xy  # The coordinates of center point on the coordinate system of feature map
        new_wh = norm_wh * torch.exp(wh)  # The width and height of bboxes on the feature map coordinate system
        new_obj = torch.sigmoid(obj)  # Map the existence of obj to 0-1
        new_cls = torch.sigmoid(cls)  # Whether the cls is correctly mapped to 0-1

        # reshape tensor -> [B, A, H, W, (C/A)]
        decode = torch.cat([new_xy, new_wh, new_obj, new_cls], dim=-1)
        decode = decode.permute(0, 3, 1, 2, 4).contiguous()  # [B, num_anchors, H, W, ...]

        return decode, norm_anchors

    def criterion(
        self,
        decode_ls: list[torch.Tensor],
        norm_anchors_ls: list[torch.Tensor],
        iid_cls_bboxes: torch.Tensor,
    ) -> tuple:
        cls_loss = torch.scalar_tensor(0, dtype=torch.float32, device=self.device)
        bbox_loss = torch.scalar_tensor(0, dtype=torch.float32, device=self.device)
        obj_loss = torch.scalar_tensor(0, dtype=torch.float32, device=self.device)

        # 使用BCELoss
        BCEcls = nn.BCELoss()
        BCEobj = nn.BCELoss()

        for i, (decode, norm_anchors) in enumerate(zip(decode_ls, norm_anchors_ls)):
            tcls, tbboxes, indices = self.prepare_target(
                decode, norm_anchors, iid_cls_bboxes
            )  # indices means that there is a bbox

            num_bboxes = tbboxes.size(0)
            tobj = torch.zeros_like(decode[..., 0], device=self.device)  # decode [B, A, H, W, (C / A)]

            if num_bboxes:
                pbox = decode[indices][:, :4]
                iou = bbox_iou(pbox, tbboxes[i], bbox_format="coco", iou_type="ciou")
                bbox_loss += (1.0 - iou).mean()  # iou loss

                tobj[indices] = iou.detach().clamp(0.0, 1.0).type(tobj.dtype)

                if decode.size(-1) > 5:
                    pcls = decode[indices][:, 5:]  # pcls [num_bboxes, 80]
                    cls = torch.zeros_like(pcls, device=self.device)  # cls [num_bboxes, 80]
                    cls[range(num_bboxes), tcls[i]] = 1.0

                    cls_loss += BCEcls(pcls, cls)

            obj_preds = decode[..., 4]
            obj_loss += BCEobj(obj_preds, tobj)

        loss_components = [bbox_loss, obj_loss, cls_loss]
        loss = self.b_weiget * bbox_loss + self.o_weiget * obj_loss + self.c_weiget * cls_loss

        return loss, loss_components

    @torch.no_grad()
    def prepare_target(
        self,
        decode: torch.Tensor,
        norm_anchors: torch.Tensor,
        iid_cls_bboxes: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        """
        The target information of the original image is mapped to the fmap space to prepare for the loss calculation

        The reason for not mapping the feature space to the original image space is that the number of bounding boxes in
        the feature space is much larger than that in the target bounding box. For example, the number of bounding boxes
        in the 52*52 feature map is 52*52*3, and the computational complexity is high.
        """
        num_bboxes = iid_cls_bboxes.shape[0]  # number of bboxes(targets)
        height, width = decode.shape[2], decode.shape[3]  # decode [B, A, H, W, (C / A)]

        targets = iid_cls_bboxes.repeat(self.anchor_nums, 1, 1)  # targets [num_anchors, num_bboxes, 6]
        targets[:, :, 2:6] *= iid_cls_bboxes.new([width, height]).repeat(num_bboxes, 2)
        anchor_ids = (
            torch.arange(self.anchor_nums, device=self.device)
            .view(self.anchor_nums, 1)
            .repeat(1, num_bboxes)
            .view(self.anchor_nums, num_bboxes, 1)
        )
        targets = torch.cat([anchor_ids, targets], dim=-1)  # (anchor_ids, img_id, class_id, x, y, w, h)

        # Filter the appropriate target box
        if num_bboxes:
            r = targets[:, :, 5:7] / norm_anchors[:, None]
            j = torch.max(r, 1.0 / r).max(2)[0] < self.anchor_scale_threshold
            targets = targets[j]

        anchor_ids, img_ids, tcls = targets[:, :3].long().T

        # The center point coordinates of the target bboxes in the feature map coordinate system
        gxy = targets[:, 3:5]
        gji = gxy.long()
        gj, gi = gji.T

        tbboxes = targets[3:7]  # box
        indices = (img_ids, anchor_ids, gi.clamp_(0, height - 1), gj.clamp_(0, width - 1))

        return tcls, tbboxes, indices


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
