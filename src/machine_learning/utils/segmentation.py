from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

import numpy as np
from PIL import ImageColor
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from machine_learning.utils.constants import CSS_COLORS


def calculate_miou(
    predictions: torch.Tensor, targets: torch.Tensor, ignore_value: int = 255
) -> tuple[float, tuple[torch.Tensor, torch.Tensor]]:
    """Calculate mean Intersection over Union (mIoU) for the batch."""
    predictions = F.interpolate(predictions, size=targets.shape[-2:], mode="bilinear", align_corners=False)

    pred_classes = torch.argmax(predictions, dim=1)

    valid_mask = targets != ignore_value

    num_classes = predictions.shape[1]
    intersection = torch.zeros(num_classes, device=predictions.device)
    union = torch.zeros(num_classes, device=predictions.device)

    for cls in range(num_classes):
        pred_mask = (pred_classes == cls) & valid_mask
        target_mask = (targets == cls) & valid_mask

        intersection[cls] = (pred_mask & target_mask).sum()
        union[cls] = (pred_mask | target_mask).sum()

    valid_classes = union > 0

    if not valid_classes.any():
        return 0.0, (intersection, union)

    iou = intersection[valid_classes] / union[valid_classes]

    return iou.mean().item(), (intersection, union)


def rescale_masks(
    masks: np.ndarray | torch.Tensor,
    img_shape: tuple[int, int],
    img0_shape: tuple[int, int],
    ratio_pad=None,
    padding=True,
):
    """
    Rescale segmentation masks from the network output shape back to the original image shape,
    perfectly handling the network's stride (spatial downsampling).

    Args:
        masks: The segmentation masks, format (N, H, W) or (H, W). Its spatial size can be smaller than img_shape due to network stride.
        img_shape: The shape of the padded image sent to the model (height, width).
        img0_shape: The original shape of the target image (height, width).
        ratio_pad: A tuple of (ratio, pad) for scaling. If not provided, it will be calculated.
        padding: If True, assuming the image has letterbox style padding that needs to be cropped.
    """
    is_numpy = isinstance(masks, np.ndarray)
    if is_numpy:
        masks = torch.from_numpy(masks)

    if ratio_pad is None:
        gain = min(img_shape[0] / img0_shape[0], img_shape[1] / img0_shape[1])
        pad = (
            (img_shape[1] - img0_shape[1] * gain) / 2,
            (img_shape[0] - img0_shape[0] * gain) / 2,
        )
    else:
        pad = ratio_pad[1]

    original_ndim = masks.ndim
    while masks.ndim < 4:
        masks = masks.unsqueeze(0)

    mh, mw = masks.shape[-2:]
    ih, iw = img_shape

    ratio_h = mh / ih
    ratio_w = mw / iw

    pad_top = pad[1] * ratio_h
    pad_left = pad[0] * ratio_w

    top, left = int(round(pad_top)), int(round(pad_left))
    bottom, right = int(round(mh - pad_top)), int(round(mw - pad_left))

    if padding:
        masks = masks[..., top:bottom, left:right]

    if masks.numel() == 0:
        empty_shape = list(masks.shape[:-2]) + list(img0_shape)
        res = torch.zeros(empty_shape, dtype=masks.dtype, device=masks.device)
        return res.numpy() if is_numpy else res

    masks = F.interpolate(masks.float(), size=img0_shape, mode="nearest")

    while masks.ndim > original_ndim:
        masks = masks.squeeze(0)

    if is_numpy:
        masks = masks.cpu().numpy()

    return masks


def visualize_mask(mask: np.ndarray) -> None:
    """
    Visualize both semantic mask [H, W] and instance/panoptic masks [N, H, W].
    """
    if mask.ndim == 2:
        H, W = mask.shape
        title_prefix = "Semantic Segmentation"
    elif mask.ndim == 3:
        N, H, W = mask.shape
        title_prefix = "Instance/Panoptic Segmentation"
    else:
        raise ValueError(f"Unsupported mask dimension: {mask.ndim}. Expected 2 or 3.")

    display_mask = np.zeros((H, W, 3), dtype=np.uint8)
    num_items = 0

    if mask.ndim == 2:
        unique_cls = np.unique(mask)
        for i, cls in enumerate(unique_cls):
            if cls != 0:
                rgb_color = ImageColor.getrgb(CSS_COLORS[num_items % len(CSS_COLORS)])
                display_mask[mask == cls] = rgb_color
                num_items += 1

    elif mask.ndim == 3:
        for i in range(N):
            instance_mask = mask[i]
            if instance_mask.max() > 0:
                rgb_color = ImageColor.getrgb(CSS_COLORS[num_items % len(CSS_COLORS)])
                display_mask[instance_mask > 0] = rgb_color
                num_items += 1

    plt.figure(figsize=(12, 12))
    plt.axis("off")
    plt.title(f"{title_prefix} - Total Foreground Items: {num_items}")
    plt.imshow(display_mask)
    plt.show()


def visualize_segmentation(
    img: torch.Tensor,
    pred_mask: torch.Tensor,
    class_maps: Sequence[str] | Mapping[int, str] | None = None,
    alpha: float = 0.5,
) -> None:
    """Visualize segmentation results by overlaying the predicted mask on the input image."""
    while pred_mask.dim() < 4:
        pred_mask = pred_mask.unsqueeze(0)

    pred_mask = F.interpolate(pred_mask.float(), size=img.shape[-2:], mode="nearest").squeeze().long()

    class_ids = torch.unique(pred_mask).cpu().tolist()

    color_map = {}
    legend_patches = []

    for cls_id in class_ids:
        cls_id = int(cls_id)
        rgb_color = ImageColor.getrgb(CSS_COLORS[cls_id % len(CSS_COLORS)])
        color_map[cls_id] = rgb_color

        if class_maps is not None:
            if isinstance(class_maps, (list, tuple)):
                label = class_maps[cls_id] if cls_id < len(class_maps) else f"Class {cls_id}"
            else:  # Mapping (dict)
                label = class_maps.get(cls_id, f"Class {cls_id}")
        else:
            label = f"Class {cls_id}"

        normalized_color = [c / 255.0 for c in rgb_color]
        patch = mpatches.Patch(color=normalized_color, label=label)
        legend_patches.append(patch)

    color_mask = torch.zeros((3, *pred_mask.shape), dtype=torch.uint8, device=pred_mask.device)
    for cls, color in color_map.items():
        color_mask[:, pred_mask == cls] = torch.tensor(color, device=pred_mask.device).view(3, 1, 1)

    plt.figure(figsize=(12, 12))
    plt.axis("off")

    display_img = img.permute(1, 2, 0).cpu().numpy()
    if display_img.dtype != "uint8" and display_img.max() > 1.0:
        display_img = display_img / 255.0

    plt.imshow(display_img)
    display_mask = color_mask.permute(1, 2, 0).cpu().numpy()
    plt.imshow(display_mask, alpha=alpha)
    plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0, fontsize=12)
    plt.tight_layout()
    plt.show()
