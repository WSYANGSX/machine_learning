import torch
import torch.nn.functional as F

import numpy as np
from matplotlib import pyplot as plt


def calculate_miou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    ignore_value: int = 255,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Optimized version of mIoU calculation (no loop required)."""
    preds = F.interpolate(preds, size=targets.shape[-2:], mode="bilinear", align_corners=False)
    pred_classes = torch.argmax(preds, dim=1)

    valid_mask = targets != ignore_value

    targets_clean = targets.clone()
    targets_clean[~valid_mask] = 0
    pred_classes_clean = pred_classes.clone()
    pred_classes_clean[~valid_mask] = 0

    num_classes = preds.shape[1]

    intersection = torch.zeros(num_classes, device=preds.device)
    union = torch.zeros(num_classes, device=preds.device)

    for cls in range(num_classes):
        pred_mask = (pred_classes_clean == cls) & valid_mask
        target_mask = (targets_clean == cls) & valid_mask
        intersection[cls] = (pred_mask & target_mask).sum().float()
        union[cls] = (pred_mask | target_mask).sum().float()

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


def generate_gt_edges(
    mask: torch.Tensor,
    edge_width: int = 3,
    exclude_value: int = 255,
) -> torch.Tensor:
    """
    Extract and thicken the edges through morphological operations.
    Strictly ensure that the dimensions of the input and output are consistent:
    - Input [H, W] -> Output [H, W]
    - Input [N, H, W] -> Output [N, H, W]

    Args:
    mask: Binary or category index mask, supporting 2D or 3D.
    edge_width: The bold width of the edge (it is recommended to be an odd number, such as 1, 3, 5)
    exclude_value: The value to ignore/exclude from edge generation (e.g., 255).

    Returns:
    thick_edges: A binary edge map with the same shape as the input.
    """
    original_dim = mask.dim()

    if original_dim == 2:
        # [H, W] -> [1, 1, H, W]
        x = mask.unsqueeze(0).unsqueeze(0)
    elif original_dim == 3:
        # [N, H, W] -> [N, 1, H, W]
        x = mask.unsqueeze(1)
    else:
        raise ValueError(f"Only 2D [H, W] or 3D [N, H, W] input is supported. The current dimension is: {original_dim}")

    x_float = x.float()

    dilated_mask = F.max_pool2d(x_float, kernel_size=3, stride=1, padding=1)
    eroded_mask = -F.max_pool2d(-x_float, kernel_size=3, stride=1, padding=1)
    base_edges = (dilated_mask != eroded_mask).float()

    valid_pixels = ((x_float > 0) & (x_float != exclude_value)).float()
    has_valid_class = F.max_pool2d(valid_pixels, kernel_size=3, stride=1, padding=1) > 0

    base_edges[~has_valid_class] = 0.0

    if edge_width > 1:
        padding = edge_width // 2
        thick_edges = F.max_pool2d(base_edges, kernel_size=edge_width, stride=1, padding=padding)
    else:
        thick_edges = base_edges

    if original_dim == 2:
        # [1, 1, H, W] -> [H, W]
        out = thick_edges.squeeze(0).squeeze(0)
    else:
        # [N, 1, H, W] -> [N, H, W]
        out = thick_edges.squeeze(1)

    return out


class SegmentMetrics:
    """
    Metrics for Semantic Segmentation Task.
    Source from: https://github.com/VainF/DeepLabV3Plus-Pytorch.
    """

    def __init__(self, nc: int) -> None:
        self.nc = nc
        self.confusion_matrix = np.zeros((nc, nc))

    def update(self, target: np.ndarray, preds: np.ndarray):
        for gt, pred in zip(target, preds):
            self.confusion_matrix += self._fast_hist(gt.flatten(), pred.flatten())

    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k != "Class IoU":
                string += "%s: %f\n" % (k, v)

        return string

    def _fast_hist(self, target: np.ndarray, preds: np.ndarray):
        mask = (target >= 0) & (target < self.nc)
        hist = np.bincount(
            self.nc * target[mask].astype(int) + preds[mask],
            minlength=self.nc**2,
        ).reshape(self.nc, self.nc)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
        - overall accuracy
        - mean accuracy
        - mean I0U
        - fwavacc
        """
        cm = self.confusion_matrix
        acc = np.diag(cm).sum() / cm.sum()
        acc_cls = np.diag(cm) / cm.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(cm) / (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm))
        mean_iu = np.nanmean(iu)
        freq = cm.sum(axis=1) / cm.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.nc), iu))

        return {
            "Overall Acc": acc,
            "Mean Acc": acc_cls,
            "FreqW Acc": fwavacc,
            "Mean IoU": mean_iu,
            "Class IoU": cls_iu,
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.nc, self.nc))
