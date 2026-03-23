from typing import Literal, Mapping, Any

import cv2
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.amp import autocast
from scipy.optimize import linear_sum_assignment
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, ToTensor, Normalize

from machine_learning.networks import BaseNet
from machine_learning.types.aliases import FilePath
from machine_learning.utils.logger import LOGGER, colorstr
from machine_learning.algorithms.base import AlgorithmBase
from machine_learning.utils.detection import pad_to_square


class MaskFormer(AlgorithmBase):
    def __init__(
        self,
        cfg: FilePath | Mapping[str, Any],
        net: BaseNet | None = None,
        name: str | None = "maskformer",
        device: Literal["cuda", "cpu", "auto"] = "auto",
        amp: bool = True,
        ema: bool = True,
        modality: str = "img",
    ) -> None:
        """
        Implementation of maskformer segmentation algorithm.

        Args:
            cfg (FilePath, Mapping[str, Any]): Configuration of the algorithm, it can be yaml file path or cfg dict.
            data (Mapping[str, Union[Dataset, Any]]): Parsed specific dataset data, must including train dataset and val
            dataset, may contain data information of the specific dataset.
            net (BaseNet): Models required by the MaskFormer algorithm.
            name (str): Name of the algorithm, it can be instantiated by cfg. Defaults to 'maskformer'.
            device (Literal["cuda", "cpu", "auto"], optional): Running device. Defaults to
            "auto"-automatic selection by algorithm.
            amp (bool): Whether to enable Automatic Mixed Precision. Defaults to False.
            ema (bool): Whether to enable Exponential Moving Average. Defaults to True.
            modality (str): The data modality to use for multimodal dataset selection. Only relevant for multimodal
            datasets. Defaults to "img".
        """
        super().__init__(cfg=cfg, net=net, name=name, device=device, amp=amp, ema=ema)

        self.modality = modality

        # main parameters of the algorithm
        self.task = self.cfg["algorithm"]["task"]
        self.imgsz = self.cfg["algorithm"]["imgsz"]
        self.close_mosaic_epoch = self.cfg["algorithm"]["close_mosaic_epoch"]
        self.single_cls = self.cfg["data"]["single_cls"]  # only one class (foreground) + background
        self.plot = self.cfg["algorithm"].get("plot", False)  # whether to plot validation results

        # maskformer specific parameters
        self.num_queries = self.cfg["algorithm"].get("num_queries", 100)

        # weights
        self.cls_weight = self.cfg["algorithm"].get("cls", 2.0)
        self.mask_weight = self.cfg["algorithm"].get("mask", 5.0)
        self.dice_weight = self.cfg["algorithm"].get("dice", 5.0)
        self.no_object_weight = self.cfg["algorithm"].get("no_object", 0.1)  # 'No object' weight

    def _init_on_trainer(self, train_cfg, dataset):
        """Initialize the datasets, dataloaders, nets, optimizers, and schedulers.
        The attributes that require the dataset parameter are created here
        """
        super()._init_on_trainer(train_cfg, dataset)

        self.nc = 1 if self.single_cls else int(self.dataset_cfg["nc"])
        self.class_names = ["object"] if self.single_cls else self.dataset_cfg["class_names"]

        self.empty_weight = torch.ones(self.nc + 1, device=self.device)
        self.empty_weight[-1] = self.no_object_weight

    def _init_on_evaluator(self, ckpt, dataset, use_dataset):
        super()._init_on_evaluator(ckpt, dataset, use_dataset)

        self.nc = 1 if self.single_cls else int(self.dataset_cfg["nc"])
        self.class_names = ["object"] if self.single_cls else self.dataset_cfg["class_names"]

    def _init_optimizers(self) -> None:
        self.opt_cfg = self._cfg["optimizer"]

        weight_decay = self.opt_cfg.get("weight_decay", 1e-5) * self.batch_size * self.accumulate / self.nbs
        iterations = (
            math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.nbs)) * self.trainer_cfg["epochs"]
        )
        self.optimizer = self.build_optimizer(
            name=self.opt_cfg.get("opt", "auto"),
            lr=self.opt_cfg.get("lr", 0.001),
            momentum=self.opt_cfg.get("momentum", 0.9),
            decay=weight_decay,
            iterations=iterations,
        )

        self._add_optimizer("optimizer", self.optimizer)

    def _init_schedulers(self) -> None:
        self.sch_config = self._cfg["scheduler"]

        if self.sch_config.get("sched") == "CustomLRDecay":
            self.lf = (
                lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.opt_cfg["final_factor"])
                + self.opt_cfg["final_factor"]
            )  # linear
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
            self._add_scheduler("scheduler", self.scheduler)

        else:
            LOGGER.warning(f"Unknown scheduler type '{self.sch_config.get('type')}', no scheduler configured.")

    def train_epoch(self, epoch: int, writer: SummaryWriter, log_interval: int = 10) -> dict[str, float]:
        super().train_epoch(epoch, writer, log_interval)

        # close mosaic
        if epoch == int(self.close_mosaic_epoch * self.epochs):
            self.close_dataloader_mosaic()

        # log metrics
        metrics = {"tloss": 0.0, "mask_loss": 0.0, "dice_loss": 0.0, "closs": 0.0, "instances": 0, "img_size": None}
        self.print_metric_titles("train", metrics)

        pbar = tqdm(enumerate(self.train_loader), total=self.train_batches)
        for i, batch in pbar:
            # Warmup
            batches = epoch * self.train_batches + i
            self.warmup(batches, epoch)

            # Load data
            imgs = batch[self.modality].to(self.device, non_blocking=True).float() / 255

            # For MaskFormer, Dataloader should provide batch["masks"] [N, H, W]
            batch_idx = batch["batch_idx"].to(self.device)
            cls_ids = batch["cls"].view(-1).to(self.device)
            masks = batch["masks"].to(self.device) if "masks" in batch else None

            # Group Ground Truths by image
            gt_labels = [cls_ids[batch_idx == j] for j in range(imgs.size(0))]
            gt_masks = [masks[batch_idx == j] for j in range(imgs.size(0))] if masks is not None else []

            # Loss calculation
            with autocast(
                device_type=str(self.device), enabled=self.amp
            ):  # Ensure that the autocast scope correctly covers the forward computation
                preds = self.net(imgs)
                loss, lc = self.criterion(preds, gt_labels, gt_masks)

            # Gradient backpropagation
            self.backward(loss)
            # Parameter optimization
            self.optimizer_step(batches)

            # Metrics
            mask_loss, closs, dice_loss = lc["mask_loss"], lc["cls_loss"], lc["dice_loss"]

            metrics["tloss"] = (metrics["tloss"] * i + loss.item()) / (i + 1)  # tloss
            metrics["mask_loss"] = (metrics["mask_loss"] * i + mask_loss) / (i + 1)
            metrics["closs"] = (metrics["closs"] * i + closs) / (i + 1)
            metrics["dice_loss"] = (metrics["dice_loss"] * i + dice_loss) / (i + 1)
            metrics["img_size"] = imgs.size(2)
            metrics["instances"] = sum(len(lbl) for lbl in gt_labels)

            if i % log_interval == 0:
                writer.add_scalar("mask_loss/train_batch", mask_loss, batches)
                writer.add_scalar("closs/train_batch", closs, batches)
                writer.add_scalar("dice_loss/train_batch", dice_loss, batches)

            # log
            self.pbar_log("train", pbar, epoch, **metrics)

        return metrics, {}

    def close_dataloader_mosaic(self) -> None:
        if hasattr(self.train_loader.dataset, "close_mosaic"):
            LOGGER.info("Closing dataloader mosaic")
            self.train_loader.dataset.close_mosaic()

    @torch.no_grad()
    def match(self, pred_logits: torch.Tensor, pred_masks: torch.Tensor, gt_labels: list, gt_masks: list):
        """Bipartite matching for MaskFormer via Hungarian algorithm."""
        indices = []
        for b in range(len(gt_labels)):
            if len(gt_labels[b]) == 0:
                indices.append((torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)))
                continue

            out_prob = pred_logits[b].softmax(-1)  # [Q, NC+1]
            out_mask = pred_masks[b]  # [Q, H, W]

            tgt_ids = gt_labels[b].long()
            tgt_mask = gt_masks[b].to(out_mask)

            # Compute cost matrices
            cost_class = -out_prob[:, tgt_ids]  # [Q, num_gt]

            out_mask_flat = out_mask.flatten(1)  # [Q, H*W]
            tgt_mask_flat = tgt_mask.flatten(1)  # [num_gt, H*W]

            # Focal loss/BCE cost and Dice cost
            device_type = str(self.device).split(":")[0]  # safely extract "cuda" or "cpu"
            with autocast(device_type=device_type, enabled=False):
                # Memory efficient mask cost calculation to prevent OOM
                cost_mask = torch.zeros((out_mask_flat.size(0), tgt_mask_flat.size(0)), device=self.device)
                for i in range(tgt_mask_flat.size(0)):
                    cost_mask[:, i] = F.binary_cross_entropy_with_logits(
                        out_mask_flat, tgt_mask_flat[i].unsqueeze(0).expand(out_mask_flat.size(0), -1), reduction="none"
                    ).mean(-1)

                # Memory efficient dice cost calculation via Matrix Multiplication
                out_mask_sig = out_mask_flat.sigmoid()
                num = 2 * (out_mask_sig @ tgt_mask_flat.T)
                den = out_mask_sig.sum(-1).unsqueeze(1) + tgt_mask_flat.sum(-1).unsqueeze(0) + 1e-8
                cost_dice = 1.0 - (num / den)

            C = self.cls_weight * cost_class + self.mask_weight * cost_mask + self.dice_weight * cost_dice
            C = C.cpu().numpy()

            row_ind, col_ind = linear_sum_assignment(C)
            indices.append((torch.as_tensor(row_ind, dtype=torch.int64), torch.as_tensor(col_ind, dtype=torch.int64)))

        return indices

    def criterion(self, preds: dict[str, torch.Tensor], gt_labels: list[torch.Tensor], gt_masks: list[torch.Tensor]):
        """MaskFormer Set Prediction Loss."""
        # Assumes BaseNet outputs a dict for MaskFormer with keys `pred_logits` and `pred_masks`
        pred_logits = preds["pred_logits"]  # [B, Q, NC+1]
        pred_masks = preds["pred_masks"]  # [B, Q, H, W]

        indices = self.match(pred_logits, pred_masks, gt_labels, gt_masks)

        # Arrange permutation
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        idx = (batch_idx, src_idx)

        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(gt_labels, indices)])
        target_classes = torch.full(pred_logits.shape[:2], self.nc, dtype=torch.int64, device=self.device)
        target_classes[idx] = target_classes_o.long()

        # 1. Classification Loss (Cross Entropy)
        loss_ce = F.cross_entropy(pred_logits.transpose(1, 2), target_classes, self.empty_weight)

        # 2. Mask Loss (BCE & Dice)
        if len(target_classes_o) == 0:
            loss_mask = pred_masks.sum() * 0
            loss_dice = pred_masks.sum() * 0
        else:
            src_masks = pred_masks[idx].flatten(1)
            target_masks_o = torch.cat([t[J] for t, (_, J) in zip(gt_masks, indices)]).to(src_masks).flatten(1)

            loss_mask = F.binary_cross_entropy_with_logits(src_masks, target_masks_o, reduction="none").mean(
                1
            ).sum() / len(target_classes_o)

            num = 2 * (src_masks.sigmoid() * target_masks_o).sum(1)
            den = src_masks.sigmoid().sum(1) + target_masks_o.sum(1) + 1e-8
            loss_dice = (1 - num / den).sum() / len(target_classes_o)

        losses = {
            "cls_loss": loss_ce.item() * self.cls_weight,
            "mask_loss": loss_mask.item() * self.mask_weight,
            "dice_loss": loss_dice.item() * self.dice_weight,
        }
        loss = loss_ce * self.cls_weight + loss_mask * self.mask_weight + loss_dice * self.dice_weight

        return loss, losses

    @torch.no_grad()
    def validate(self):
        super().validate()

        metrics = {
            "class": "all",
            "images": 0,
            "vloss": 0.0,
            "labels": 0,
            "mIoU": 0.0,
            "PQ": 0.0,
            "save_best": 0.0,
        }
        info = {}
        self.print_metric_titles("val", metrics)

        conf_thres_val = self.cfg["algorithm"].get("conf_thres_val", 0.5)
        mask_thres = self.cfg["algorithm"].get("mask_thres", 0.5)

        total_inter = torch.zeros(self.nc, device=self.device)
        total_union = torch.zeros(self.nc, device=self.device)
        pq_iou_sum = 0.0
        pq_tp = 0
        pq_fp = 0
        pq_fn = 0

        pbar = tqdm(enumerate(self.val_loader), total=self.val_batches)
        for i, batch in pbar:
            imgs = batch[self.modality].to(self.device, non_blocking=True).float() / 255
            batch_idx = batch["batch_idx"].to(self.device)
            cls_ids = batch["cls"].view(-1).to(self.device)
            masks = batch["masks"].to(self.device) if "masks" in batch else None

            gt_labels = [cls_ids[batch_idx == j] for j in range(imgs.size(0))]
            gt_masks = [masks[batch_idx == j] for j in range(imgs.size(0))] if masks is not None else []

            preds = self.net(imgs)
            loss, _ = self.criterion(preds, gt_labels, gt_masks)
            metrics["vloss"] = (metrics["vloss"] * i + loss.item()) / (i + 1)

            pred_logits = preds["pred_logits"]  # [B, Q, NC+1]
            pred_masks = preds["pred_masks"]  # [B, Q, H, W]
            probs = pred_logits.softmax(-1)

            for b in range(imgs.size(0)):
                metrics["images"] += 1
                tcls = gt_labels[b]
                tmasks = gt_masks[b] if gt_masks else torch.zeros(0, *pred_masks.shape[-2:], device=self.device)
                metrics["labels"] += int(tcls.numel())

                # Semantic prediction from queries
                class_probs = probs[b, :, : self.nc]
                mask_probs = pred_masks[b].sigmoid()
                sem_scores = torch.einsum("qc,qhw->chw", class_probs, mask_probs)
                pred_sem = sem_scores.argmax(0)

                # Build GT semantic map (ignore background)
                if tcls.numel() > 0:
                    gt_sem = torch.full_like(pred_sem, -1, dtype=torch.long)
                    for gi in range(tcls.numel()):
                        m = tmasks[gi] > 0.5
                        gt_sem[m] = tcls[gi].long()
                else:
                    gt_sem = torch.full_like(pred_sem, -1, dtype=torch.long)

                # mIoU accumulation (ignore pixels without GT)
                valid = gt_sem >= 0
                if valid.any():
                    pred_flat = pred_sem[valid].view(-1)
                    gt_flat = gt_sem[valid].view(-1)
                    conf = torch.bincount(self.nc * gt_flat + pred_flat, minlength=self.nc * self.nc).reshape(
                        self.nc, self.nc
                    )
                    inter = conf.diag()
                    union = conf.sum(0) + conf.sum(1) - inter
                    total_inter += inter
                    total_union += union

                # Panoptic Quality (PQ) via instance matching per class
                labels_all = probs[b].argmax(-1)
                class_scores, class_labels = class_probs.max(-1)
                keep = (labels_all != self.nc) & (class_scores > conf_thres_val)
                pred_inst_masks = mask_probs[keep] > mask_thres
                pred_inst_labels = class_labels[keep]

                gt_inst_masks = (tmasks > 0.5) if tcls.numel() > 0 else tmasks
                gt_inst_labels = tcls

                for c in range(self.nc):
                    p_idx = pred_inst_labels == c
                    g_idx = gt_inst_labels == c
                    p_masks = pred_inst_masks[p_idx]
                    g_masks = gt_inst_masks[g_idx]

                    if p_masks.numel() == 0 and g_masks.numel() == 0:
                        continue
                    if p_masks.numel() == 0:
                        pq_fn += int(g_masks.shape[0])
                        continue
                    if g_masks.numel() == 0:
                        pq_fp += int(p_masks.shape[0])
                        continue

                    p_flat = p_masks.flatten(1).float()
                    g_flat = g_masks.flatten(1).float()
                    inter = p_flat @ g_flat.T
                    union = p_flat.sum(1, keepdim=True) + g_flat.sum(1, keepdim=True).T - inter
                    iou = torch.where(union > 0, inter / union, torch.zeros_like(union))

                    iou_np = iou.cpu().numpy()
                    row_ind, col_ind = linear_sum_assignment(1 - iou_np)
                    matched = [(r, cidx) for r, cidx in zip(row_ind, col_ind) if iou_np[r, cidx] > 0.5]

                    tp = len(matched)
                    pq_tp += tp
                    pq_fp += int(p_masks.shape[0] - tp)
                    pq_fn += int(g_masks.shape[0] - tp)
                    if tp:
                        pq_iou_sum += float(sum(iou_np[r, cidx] for r, cidx in matched))

            denom = total_union > 0
            if denom.any():
                metrics["mIoU"] = (total_inter[denom] / total_union[denom]).mean().item()
            else:
                metrics["mIoU"] = 0.0

            pq_denom = pq_tp + 0.5 * pq_fp + 0.5 * pq_fn
            metrics["PQ"] = float(pq_iou_sum / pq_denom) if pq_denom > 0 else 0.0
            metrics["save_best"] = metrics["mIoU"]

            self.pbar_log("val", pbar, **metrics)

        # Per-class IoU info
        denom = total_union > 0
        if denom.any():
            iou_per_class = (total_inter / torch.clamp(total_union, min=1)).detach().cpu().numpy().tolist()
            info["iou_per_class"] = {
                (self.class_names[idx] if hasattr(self, "class_names") and self.class_names else f"Class {idx}"): float(
                    iou_per_class[idx]
                )
                for idx in range(self.nc)
            }

        return metrics, info

    @torch.no_grad()
    def eval(
        self,
        img_path: str | FilePath,
        conf_thres: float | None = None,
        *args,
        **kwargs,
    ) -> None:
        self.set_eval()

        # read image
        img0 = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        h0, w0, _ = img0.shape

        # scale to square
        padded_img = pad_to_square(img=img0, pad_values=0)

        # to tensor / normalize
        tfs = Compose([ToTensor(), Normalize(mean=[0, 0, 0], std=[1, 1, 1])])
        img = tfs(padded_img)
        img = img.unsqueeze(0).to(self.device)

        # input image to model
        preds = self.net(img)

        pred_logits = preds["pred_logits"][0]
        pred_masks = preds["pred_masks"][0]
        probs = pred_logits.softmax(-1)
        class_probs = probs[:, : self.nc]
        class_scores, class_labels = class_probs.max(-1)
        labels_all = probs.argmax(-1)

        conf_thres = conf_thres if conf_thres is not None else self.cfg["algorithm"].get("conf_thres_val", 0.5)
        keep = (labels_all != self.nc) & (class_scores > conf_thres)
        keep_indices = torch.nonzero(keep).squeeze(1)

        rendered = img0.copy()

        if keep_indices.numel() == 0:
            LOGGER.info("No objects detected above the confidence threshold.")
        else:
            # Iterate and render all valid predicted instances
            for idx in keep_indices:
                pred_cls = int(class_labels[idx].item())
                pred_conf = float(class_scores[idx].item())
                prob_mask = pred_masks[idx].sigmoid()

                if prob_mask.shape != img.shape[-2:]:
                    prob_mask = (
                        F.interpolate(
                            prob_mask.unsqueeze(0).unsqueeze(0),
                            size=img.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                        .squeeze(0)
                        .squeeze(0)
                    )

                prob_mask = prob_mask[:h0, :w0].detach().cpu().numpy()

                # Assign random color for each detected instance
                color = np.random.randint(0, 255, (1, 1, 3), dtype=np.uint8)
                mask_bool = prob_mask > self.cfg["algorithm"].get("mask_thres", 0.5)

                colored_mask = np.zeros_like(img0)
                colored_mask[mask_bool] = color
                rendered = cv2.addWeighted(rendered, 1.0, colored_mask, 0.5, 0.0)

                class_name = (
                    self.class_names[pred_cls] if hasattr(self, "class_names") and self.class_names else str(pred_cls)
                )

                y_indices, x_indices = np.where(mask_bool)
                if len(y_indices) > 0:
                    cy, cx = int(np.mean(y_indices)), int(np.mean(x_indices))
                    cv2.putText(
                        rendered,
                        f"{class_name}: {pred_conf:.2f}",
                        (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
        cv2.imshow("maskformer_inference", cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyWindow("maskformer_inference")

    def warmup(self, batch_inters: int, epoch: int) -> int:
        wb = (
            max(round(self.opt_cfg["warmup_epochs"] * self.train_batches), 100)
            if self.opt_cfg["warmup_epochs"] > 0
            else -1
        )  # warmup batches

        if batch_inters <= wb:
            xi = [0, wb]  # x interp
            self.accumulate = max(1, int(np.interp(batch_inters, xi, [1, self.nbs / self.batch_size]).round()))
            for j, x in enumerate(self.optimizer.param_groups):
                # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                x["lr"] = np.interp(
                    batch_inters,
                    xi,
                    [self.opt_cfg["warmup_bias_lr"] if j == 2 else 0.0, x["initial_lr"] * self.lf(epoch)],
                )
                if "momentum" in x:
                    x["momentum"] = np.interp(
                        batch_inters, xi, [self.opt_cfg["warmup_momentum"], self.opt_cfg["momentum"]]
                    )

    def build_optimizer(self, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """
        Constructs an optimizer for the given model, based on the specified optimizer name, learning rate, momentum,
        weight decay, and number of iterations.

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected
                based on the number of iterations. Default: 'auto'.
            lr (float, optional): The learning rate for the optimizer. Default: 0.001.
            momentum (float, optional): The momentum factor for the optimizer. Default: 0.9.
            decay (float, optional): The weight decay for the optimizer. Default: 1e-5.
            iterations (float, optional): The number of iterations, which determines the optimizer if
                name is 'auto'. Default: 1e5.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        """
        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        if name == "auto":
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                f"ignoring 'lr={self.opt_cfg['lr']}' and 'momentum={self.opt_cfg['momentum']}' and "
                f"determining best 'optimizer', 'lr' and 'momentum' automatically... "
            )
            lr_fit = round(0.002 * 5 / (4 + self.dataset_cfg["nc"]), 6)  # lr0 fit equation to 6 decimal places
            name, lr, momentum = ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            self.opt_cfg["warmup_bias_lr"] = 0.0  # no higher than 0.01 for Adam

        for module_name, module in self.net.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn):  # weight (no decay)
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

        optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "auto"}
        name = {x.lower(): x for x in optimizers}.get(name.lower())
        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optimizer = getattr(torch.optim, name, torch.optim.Adam)(
                g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0
            )
        elif name == "RMSProp":
            optimizer = torch.optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == "SGD":
            optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers {optimizers}. "
                "Request support for addition optimizers at https://github.com/ultralytics/ultralytics."
            )

        optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f"{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)"
        )

        return optimizer


"""
Helper functions
"""
