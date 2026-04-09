import torch
import torch.nn.functional as F

import numpy as np
from PIL import ImageColor
from matplotlib import pyplot as plt
from machine_learning.utils.constants import CSS_COLORS


def calculate_miou(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    ignore_value: int = -100,
) -> tuple[float, torch.Tensor, torch.Tensor]:
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

    return iou.mean().item(), intersection, union


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
        masks: The segmentation masks with (N, H, W) or (H, W). Its can be smaller than img_shape due to network stride.
        img_shape: The shape of the padded image sent to the model (height, width).
        img0_shape: The original shape of the target image (height, width).
        ratio_pad: A tuple of (ratio, pad) for scaling. If not provided, it will be calculated.
        padding: If True, assuming the image has letterbox style padding that needs to be cropped.
    """
    is_numpy = isinstance(masks, np.ndarray)
    if is_numpy:
        masks = torch.from_numpy(masks)

    orig_dtype = masks.dtype
    orig_ndim = masks.ndim

    if ratio_pad is None:
        gain = min(img_shape[0] / img0_shape[0], img_shape[1] / img0_shape[1])
        pad = (
            (img_shape[1] - img0_shape[1] * gain) / 2,
            (img_shape[0] - img0_shape[0] * gain) / 2,
        )  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    # net strides
    mh, mw = masks.shape[-2:]
    ih, iw = img_shape
    stride_h = ih / mh
    stride_w = iw / mw

    # masks pad
    pad_top = pad[1] / stride_h
    pad_left = pad[0] / stride_w
    top, left = int(round(pad_top)), int(round(pad_left))
    bottom, right = int(round(mh - pad_top)), int(round(mw - pad_left))

    if padding:
        masks = masks[..., top:bottom, left:right]

    if orig_ndim == 2:  # (H, W) -> (1, 1, H, W)
        masks = masks.unsqueeze(0).unsqueeze(0)
    elif orig_ndim == 3:  # (N, H, W) -> (N, 1, H, W)
        masks = masks.unsqueeze(1)
    else:
        raise ValueError(f"Unsupported mask dimensions: {orig_ndim}")

    masks = F.interpolate(masks.float(), size=img0_shape, mode="nearest")

    if orig_ndim == 2:
        masks = masks.squeeze(0).squeeze(0)
    elif orig_ndim == 3:
        masks = masks.squeeze(1)

    masks = masks.to(orig_dtype)

    return masks.numpy() if is_numpy else masks


def generate_distinct_color(id_val: int) -> tuple[int, int, int]:
    """
    Based on the PASCAL VOC displacement algorithm, generate RGB colors with extremely significant visual differences
    for any ID. Ensure that consecutive ids can also achieve extremely contrasting colors.
    """
    r, g, b = 0, 0, 0
    for i in range(8):
        if id_val == 0:
            break
        r |= (id_val & 1) << (7 - i)
        g |= ((id_val >> 1) & 1) << (7 - i)
        b |= ((id_val >> 2) & 1) << (7 - i)
        id_val >>= 3
    return (r, g, b)


def colour_mask(mask: np.ndarray) -> tuple:
    """
    Colour both semantic mask [H, W] and instance/panoptic masks [N, H, W].
    """
    if mask.ndim == 2:
        H, W = mask.shape
    elif mask.ndim == 3:
        N, H, W = mask.shape
    else:
        raise ValueError(f"Unsupported mask dimension: {mask.ndim}. Expected 2 or 3.")

    display_mask = np.zeros((H, W, 3), dtype=np.uint8)
    num_items = 0

    if mask.ndim == 2:
        unique_cls = np.unique(mask)
        for cls in unique_cls:
            if cls != 0:
                rgb_color = generate_distinct_color(int(cls))
                display_mask[mask == cls] = rgb_color
                num_items += 1

    elif mask.ndim == 3:
        for i in range(N):
            instance_mask = mask[i]
            if instance_mask.max() > 0:
                num_items += 1
                rgb_color = generate_distinct_color(num_items)
                display_mask[instance_mask > 0] = rgb_color

    return display_mask, num_items


def visualize_mask(mask: np.ndarray) -> None:
    """
    Visualize both semantic mask [H, W] and instance/panoptic masks [N, H, W] using matplotlib.
    """
    title_prefix = "Semantic" if mask.ndim == 2 else "Instance/Panoptic"
    display_mask, num_items = colour_mask(mask)

    plt.figure(figsize=(12, 12))
    plt.axis("off")
    plt.title(f"{title_prefix} Segmentation - Foreground Items: {num_items}")
    plt.imshow(display_mask)
    plt.show()
