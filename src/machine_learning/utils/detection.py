from typing import Union, Sequence, Mapping, Any

import cv2
import math
import time
import torch
import numpy as np
from copy import deepcopy
import torch.nn.functional as F
from matplotlib import pyplot as plt

from machine_learning.utils.logger import LOGGER
from machine_learning.utils.ops import zeros_like


def xywh2xyxy(x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    new = zeros_like(x)
    new[..., 0] = x[..., 0] - x[..., 2] / 2
    new[..., 1] = x[..., 1] - x[..., 3] / 2
    new[..., 2] = x[..., 0] + x[..., 2] / 2
    new[..., 3] = x[..., 1] + x[..., 3] / 2

    return new


def xyxy2xywh(x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    new = zeros_like(x)
    new[..., 0] = (x[..., 0] + x[..., 2]) / 2
    new[..., 1] = (x[..., 1] + x[..., 3]) / 2
    new[..., 2] = x[..., 2] - x[..., 0]
    new[..., 3] = x[..., 3] - x[..., 1]

    return new


def yolo2voc(bboxes: Union[torch.Tensor, np.ndarray], img_w: int, img_h: int) -> Union[torch.Tensor, np.ndarray]:
    bboxes = xywh2xyxy(bboxes)
    bboxes[..., [0, 2]] *= img_w
    bboxes[..., [1, 3]] *= img_h
    return bboxes


def pad_to_square(
    img: Union[torch.Tensor, np.ndarray], pad_values: Union[int, tuple] = 0
) -> Union[torch.Tensor, np.ndarray]:
    # first to convert to np
    output_type = "np"
    if isinstance(img, torch.Tensor):
        output_type = "torch"
        dtype = img.dtype

        if img.ndim == 3:
            img = np.array(img.permute(1, 2, 0))
        elif img.ndim == 2:
            img = np.array(img)
        else:
            raise ValueError(f"Unsupported tensor shape: {img.shape}. Expected 2D (H,W) or 3D (C,H,W).")

    h, w = img.shape[0], img.shape[1]
    dim_diff = np.abs(h - w)
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2

    if img.ndim == 2:  # gray (H, W)
        pad_width = [(pad1, pad2), (0, 0)] if h <= w else [(0, 0), (pad1, pad2)]
    else:  # rgb (H, W, C)
        pad_width = [(pad1, pad2), (0, 0), (0, 0)] if h <= w else [(0, 0), (pad1, pad2), (0, 0)]

    if isinstance(pad_values, int):
        constant_values = pad_values
    else:
        if isinstance(pad_values, tuple):
            if img.ndim == 3:  # RGB
                if len(pad_values) == 1:
                    pad_values = pad_values * img.shape[2]
                elif len(pad_values) != img.shape[2]:
                    raise ValueError(f"Padded values must have length 1 or {img.shape[2]} for RGB images.")
                constant_values = [(v, v) for v in pad_values]  # (left, right) for each channel
            elif img.ndim == 2:  # gray
                if len(pad_values) != 1:
                    raise ValueError("Padded values must be a single int or a 1-tuple for grayscale images.")
                constant_values = pad_values[0]
        else:
            raise ValueError("Padded values must be an int or a tuple.")

    output = np.pad(img, pad_width, "constant", constant_values=constant_values)

    if output_type == "torch":
        output = torch.tensor(output, dtype=dtype)
        if output.ndim == 3:
            output = output.permute(2, 0, 1)

    return output


def resize(img: Union[torch.Tensor, np.ndarray], size: int) -> Union[torch.Tensor, np.ndarray]:
    if isinstance(img, torch.Tensor):
        output_type = "torch"
        original_dtype = img.dtype
        if original_dtype != torch.float32 and original_dtype != torch.float64:
            img = img.float()
    else:
        output_type = "np"
        original_dtype = img.dtype
        original_shape = img.shape

        if img.ndim == 3:
            img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        elif img.ndim == 2:
            img = torch.from_numpy(img).float().unsqueeze(0)
        else:
            raise ValueError(f"Unsupported array shape: {img.shape}. Expected 2D (H, W) or 3D (H, W, C).")

    if img.dim() == 3:  # CHW
        img = F.interpolate(img.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    elif img.dim() == 2:  # HW
        img = F.interpolate(img.unsqueeze(0).unsqueeze(0), size=size, mode="nearest").squeeze(0).squeeze(0)
    else:
        raise ValueError(f"Unsupported tensor dimension: {img.dim()}")

    if output_type == "np":
        img = img.numpy()

        if original_dtype != np.float32 and original_dtype != np.float64:
            if original_dtype == np.uint8:
                img = np.clip(img, 0, 255)
            img = img.astype(original_dtype)

        if len(original_shape) == 3 and original_shape[2] < original_shape[0]:
            img = img.transpose(1, 2, 0)

    elif output_type == "torch":
        if original_dtype != torch.float32 and original_dtype != torch.float64:
            img = img.to(original_dtype)

    return img


def rescale_bboxes(
    bboxes: Union[torch.Tensor, np.ndarray], img_size: int, org_img_w: int, org_img_h: int
) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert the bounding box coordinates output by the object detection model from the coordinate system of the padded
    square image back to the coordinate system of the original image.
    """
    # calculate the increased pad and deal with the situation of expansion and contraction after the pad
    pad_x = max(org_img_h - org_img_w, 0) * (img_size / max(org_img_w, org_img_h))
    pad_y = max(org_img_w - org_img_h, 0) * (img_size / max(org_img_w, org_img_h))

    # the size after removing the pad
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x

    # remap the bounding box
    bboxes[:, 0] = ((bboxes[:, 0] - pad_x // 2) / unpad_w) * org_img_w
    bboxes[:, 1] = ((bboxes[:, 1] - pad_y // 2) / unpad_h) * org_img_h
    bboxes[:, 2] = ((bboxes[:, 2] - pad_x // 2) / unpad_w) * org_img_w
    bboxes[:, 3] = ((bboxes[:, 3] - pad_y // 2) / unpad_h) * org_img_h

    return bboxes


def box_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Calculate intersection-over-union (IoU) of boxes. Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py.

    Args:
        box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes.
        box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.
    """
    # NOTE: Need .float() to get accurate iou values
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.float().unsqueeze(1).chunk(2, 2), box2.float().unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def bbox_iou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    xywh: bool = True,
    GIoU: bool = False,
    DIoU: bool = False,
    CIoU: bool = False,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Calculates the Intersection over Union (IoU) between bounding boxes.

    This function supports various shapes for `box1` and `box2` as long as the last dimension is 4.
    For instance, you may pass tensors shaped like (4,), (N, 4), (B, N, 4), or (B, N, 1, 4).
    Internally, the code will split the last dimension into (x, y, w, h) if `xywh=True`,
    or (x1, y1, x2, y2) if `xywh=False`.

    Args:
        box1 (torch.Tensor): A tensor representing one or more bounding boxes, with the last dimension being 4.
        box2 (torch.Tensor): A tensor representing one or more bounding boxes, with the last dimension being 4.
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in
                               (x1, y1, x2, y2) format. Defaults to True.
        GIoU (bool, optional): If True, calculate Generalized IoU. Defaults to False.
        DIoU (bool, optional): If True, calculate Distance IoU. Defaults to False.
        CIoU (bool, optional): If True, calculate Complete IoU. Defaults to False.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
    """
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw.pow(2) + ch.pow(2) + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
            ) / 4  # center dist**2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


def _get_covariance_matrix(boxes: torch.Tensor) -> torch.Tensor:
    """
    Generating covariance matrix from obbs.

    Args:
        boxes (torch.Tensor): A tensor of shape (N, 5) representing rotated bounding boxes, with xywhr format.

    Returns:
        (torch.Tensor): Covariance matrices corresponding to original rotated bounding boxes.
    """
    # Gaussian bounding boxes, ignore the center points (the first two columns) because they are not needed here.
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    cos = c.cos()
    sin = c.sin()
    cos2 = cos.pow(2)
    sin2 = sin.pow(2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin


def batch_probiou(obb1: torch.Tensor | np.ndarray, obb2: torch.Tensor | np.ndarray, eps: float = 1e-7) -> torch.Tensor:
    """
    Calculate the prob IoU between oriented bounding boxes, https://arxiv.org/pdf/2106.06072v1.pdf.

    Args:
        obb1 (torch.Tensor | np.ndarray): A tensor of shape (N, 5) representing ground truth obbs, with xywhr format.
        obb2 (torch.Tensor | np.ndarray): A tensor of shape (M, 5) representing predicted obbs, with xywhr format.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing obb similarities.
    """
    obb1 = torch.from_numpy(obb1) if isinstance(obb1, np.ndarray) else obb1
    obb2 = torch.from_numpy(obb2) if isinstance(obb2, np.ndarray) else obb2

    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))

    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    return 1 - hd


def nms_rotated(boxes: torch.Tensor, scores: torch.Tensor, threshold: float | None = 0.45):
    """
    NMS for oriented bounding boxes using probiou and fast-nms.

    Args:
        boxes (torch.Tensor): Rotated bounding boxes, shape (N, 5), format xywhr.
        scores (torch.Tensor): Confidence scores, shape (N,).
        threshold (float, optional): IoU threshold. Defaults to 0.45.

    Returns:
        (torch.Tensor): Indices of boxes to keep after NMS.
    """
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int8)
    sorted_idx = torch.argsort(scores, descending=True)
    boxes = boxes[sorted_idx]
    ious = batch_probiou(boxes, boxes).triu_(diagonal=1)
    pick = torch.nonzero(ious.max(dim=0)[0] < threshold).squeeze_(-1)
    return sorted_idx[pick]


def non_max_suppression(
    predictions: torch.Tensor,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes: list[int] = None,
    agnostic: bool = False,
    multi_label: bool = False,
    labels: list[list[Union[int, float, torch.Tensor]]] = (),
    max_det: int = 300,
    nc: int = 0,  # number of classes (optional)
    max_time_img: float = 0.05,
    max_nms: int = 30000,
    max_wh: int = 7680,
    in_place: bool = True,
    rotated: bool = False,
) -> list[torch.Tensor]:
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

    Args:
        predictions (torch.Tensor): A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes, and all
            classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
            list contains the apriori labels for a given image. The list should be in the format
            output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int, optional): The number of classes output by the model. Any indices after this will be considered masks.
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes into torchvision.ops.nms().
        max_wh (int): The maximum box width and height in pixels.
        in_place (bool): If True, the input predictions tensor will be modified in place.
        rotated (bool): If Oriented Bounding Boxes (OBB) are being passed for NMS.

    Returns:
        (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """
    import torchvision  # scope for faster 'import ultralytics'

    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(predictions, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        predictions = predictions[0]  # select only inference output
    if classes is not None:
        classes = torch.tensor(classes, device=predictions.device)

    if predictions.shape[-1] == 6:  # end-to-end model (BNC, i.e. 1,300,6)
        output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in predictions]
        if classes is not None:
            output = [pred[(pred[:, 5:6] == classes).any(1)] for pred in output]
        return output

    bs = predictions.shape[0]  # batch size (BCN, i.e. 1,84,6300)
    nc = nc or (predictions.shape[1] - 4)  # number of classes
    nm = predictions.shape[1] - nc - 4  # number of masks
    mi = 4 + nc  # mask start index
    xc = predictions[:, 4:mi].amax(1) > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 2.0 + max_time_img * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    predictions = predictions.transpose(-1, -2)  # shape(bs,no,sum(h*w)) to shape(bs,sum(h*w),no)
    if not rotated:
        if in_place:
            predictions[..., :4] = xywh2xyxy(predictions[..., :4])  # xywh to xyxy
        else:
            predictions = torch.cat((xywh2xyxy(predictions[..., :4]), predictions[..., 4:]), dim=-1)  # xywh to xyxy

    t = time.time()
    output = [torch.zeros((0, 6 + nm), device=predictions.device)] * bs
    for xi, x in enumerate(predictions):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]) and not rotated:
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == classes).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        scores = x[:, 4]  # scores
        if rotated:
            boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)  # xywhr
            i = nms_rotated(boxes, scores, iou_thres)
        else:
            boxes = x[:, :4] + c  # boxes (offset by class)
            # iou thres is a threshold used to determine which boxes should be removed.
            # When the IoU of two boxes exceeds this threshold, the box with the lower score will be removed
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            LOGGER.warning(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded

    return output  # [(num_kept_boxes, 6 + num_masks)]*bs


def match_predictions(
    pred_classes: torch.Tensor,
    true_classes: torch.Tensor,
    iou: torch.Tensor,
    iouv: torch.Tensor,
    use_scipy: bool = False,
) -> torch.Tensor:
    """
    Matches predictions to ground truth objects (pred_classes, true_classes) using IoU.

    Args:
        pred_classes (torch.Tensor): Predicted class indices of shape(N,).
        true_classes (torch.Tensor): Target class indices of shape(M,).
        iou (torch.Tensor): An MxN tensor containing the pairwise IoU values for predictions to ground of truth.
        iouv (torch.Tensor): IoU vector for mAP@0.5:0.95.
        use_scipy (bool): Whether to use scipy for matching (more precise).

    Returns:
        (torch.Tensor): Correct tensor of shape(N,10) for 10 IoU thresholds.
    """
    # Dx10 matrix, where D - detections, 10 - IoU thresholds
    correct = np.zeros((pred_classes.shape[0], iouv.shape[0])).astype(bool)
    # LxD matrix where L - labels (rows), D - detections (columns)
    correct_class = true_classes[:, None] == pred_classes
    iou = iou * correct_class  # zero out the wrong classes
    iou = iou.cpu().numpy()
    for i, threshold in enumerate(iouv.cpu().tolist()):
        if use_scipy:
            # WARNING: known issue that reduces mAP in https://github.com/ultralytics/ultralytics/pull/4708
            import scipy  # scope import to avoid importing for all commands

            cost_matrix = iou * (iou >= threshold)
            if cost_matrix.any():
                labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix)
                valid = cost_matrix[labels_idx, detections_idx] > 0
                if valid.any():
                    correct[detections_idx[valid], i] = True
        else:
            matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
            matches = np.array(matches).T
            if matches.shape[0]:
                if matches.shape[0] > 1:
                    matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                    matches = matches[
                        np.unique(matches[:, 1], return_index=True)[1]
                    ]  # Filter out the duplicate prediction boxes
                    # matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[
                        np.unique(matches[:, 0], return_index=True)[1]
                    ]  # Filter out the duplicate gt boxes
                correct[matches[:, 1].astype(int), i] = True  # mark as prediction correct
    return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)


def class_maps(classes: list[str]) -> dict[int, str]:
    """Create a category mapping from the category list.
    Args:
        classes (list[str]): The class names list.

    Returns:
        dict[int, str]: The mapping from class ids to names.
    """
    return {i: classes[i] for i in range(len(classes))}


def add_bbox(
    img: np.ndarray,
    bbox: np.ndarray,
    class_name: str | None = None,
    color: tuple[int] = (255, 0, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Add a single bounding box with class name to the image.

    Args:
        img (np.ndarray): The image to which a bounding box is to be added.
        bbox (np.ndarray): The bounding box parameters with voc format (x_min, y_min, x_max, y_max).
        class_name (str): The category name of the object in the bounding box.
        color (tuple[int]): The color of the bounding box. Default to red.
        thickness (int): The thickness of the bounding box. Default to 2.

    Returns:
        np.ndarray: The image with bounding box.
    """
    img_copy = deepcopy(img)
    h, w = img_copy.shape[:2]
    x_min, y_min, x_max, y_max = map(int, bbox)

    # Add bbox
    cv2.rectangle(img=img_copy, pt1=(x_min, y_min), pt2=(x_max, y_max), color=color, thickness=thickness)

    if class_name is not None:
        # Obtain the text size
        ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        label_height = int(1.3 * text_height)

        # Intelligently adjust the label position (to prevent exceeding the image boundary)
        if y_min - label_height < 0:  # Insufficient space at the top
            # Place the label at the bottom inside the bounding box
            label_y_top = y_min
            label_y_bottom = label_y_top + label_height
            text_y = label_y_top + label_height - int(0.3 * text_height)
        else:  # Normal condition
            # Place the label at the top of the bounding box
            label_y_top = y_min - label_height
            label_y_bottom = y_min
            text_y = y_min - int(0.3 * text_height)

        # Make sure the label does not extend beyond the bottom of the image
        if label_y_bottom > h:
            label_y_top = max(0, h - label_height)
            label_y_bottom = h
            text_y = label_y_bottom - int(0.3 * text_height)

        # Draw the label background and text
        cv2.rectangle(img_copy, (x_min, label_y_top), (x_min + text_width, label_y_bottom), color, -1)
        cv2.putText(
            img_copy,
            text=class_name,
            org=(x_min, text_y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.35,
            color=(255, 255, 255),
            lineType=cv2.LINE_AA,
        )

    return img_copy


def visualize_img_bboxes(
    img: np.ndarray,
    bboxes: np.ndarray,
    class_ids: np.ndarray | Sequence[int] | None = None,
    class_maps: Sequence[str] | Mapping[int, str] | None = None,
    color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
    cmap: str | None = None,
) -> None:
    """Plot the image with bounding boxes.

    Args:
        img (np.ndarray): The image to which bounding boxes are to be added.
        bboxes (np.ndarray): The Bounding boxes parameters with voc format (x_min, y_min, x_max, y_max).
        class_ids (Sequence[int] | np.ndarray): The class numbers of objects in the bounding box.
        class_maps (Sequence[str] | Mapping[int, str]): The names corresponding to the class numbers of objects.
        color (tuple[int, int, int]): The color of the bboxes. Default to (255, 0, 0).
        thickness (int): The thickness of the bboxes lines. Default to 2.
        cmap (str): Color map. Grayscale image: cmap='gray' or cmap='Greys', heatmap: cmap='hot',
        rainbow image: cmap='rainbow', blue-green gradient: cmap='viridis' (default), reversed color: Add r after any
        color mapping, such as cmap='viridis r'.
    """
    if class_ids is None:
        for bbox in bboxes:
            img = add_bbox(img=img, bbox=bbox, color=color, thickness=thickness)
    else:
        if len(class_ids) != len(bboxes):
            raise ValueError("The length of bboxes and class_ids must be the same.")

        if class_maps is not None:
            for bbox, class_id in zip(bboxes, class_ids):
                class_name = class_maps[class_id]
                img = add_bbox(img=img, bbox=bbox, class_name=class_name, color=color, thickness=thickness)
        else:
            class_maps = {i: str(i) for i in class_ids}
            for bbox, class_id in zip(bboxes, class_ids):
                class_name = class_maps[class_id]
                img = add_bbox(img=img, bbox=bbox, class_name=class_name, color=color, thickness=thickness)

    plt.figure(figsize=(12, 12))
    plt.axis("off")
    plt.imshow(img, cmap)
    plt.show()


def compute_ap(
    class_names: Sequence[str],
    tp: torch.Tensor,  # [n_preds, n_iou_thres]，0/1
    conf: torch.Tensor,  # [n_preds]
    pred_cls: torch.Tensor,  # [n_preds]
    target_cls: torch.Tensor,  # [n_targets]
    iouv: Union[torch.Tensor, Sequence[float]],
):
    """Calculate AP / mAP (close to Ultralytics implementation)."""

    def to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    tp = to_np(tp)
    conf = to_np(conf)
    pred_cls = to_np(pred_cls)
    target_cls = to_np(target_cls)
    iouv = to_np(np.array(iouv))

    nc = len(class_names)
    n_iou = len(iouv)
    ap = np.zeros((nc, n_iou), dtype=np.float32)

    # First, sort globally by confidence level
    order = np.argsort(-conf)
    tp, conf, pred_cls = tp[order], conf[order], pred_cls[order]

    results = {}
    valid_classes = []  # Classes with ground truth

    eps = 1e-16

    for ci in range(nc):
        # Predictions for the current class
        pred_mask = pred_cls == ci
        if not pred_mask.any():
            continue

        # Number of ground truths for the current class
        n_gt = (target_cls == ci).sum()
        if n_gt == 0:
            # Classes without ground truth are usually not included in the average for AP calculation
            continue

        valid_classes.append(ci)

        for iou_idx in range(n_iou):
            tp_i = tp[pred_mask, iou_idx]

            fpc = (1.0 - tp_i).cumsum()
            tpc = tp_i.cumsum()

            recall_i = tpc / (n_gt + eps)
            precision_i = tpc / (tpc + fpc + eps)

            ap[ci, iou_idx] = compute_ap_single(recall_i, precision_i)

    # No valid classes case
    if len(valid_classes) == 0:
        results["mAP"] = 0.0
        for iou_idx, iou_val in enumerate(iouv):
            results[f"mAP_{iou_val * 100:.0f}"] = 0.0
        for ci, name in enumerate(class_names):
            results[f"AP_{name}"] = 0.0
        results["precision"] = 0.0
        results["recall"] = 0.0
        return results

    valid_classes = np.array(valid_classes)

    # Compute mAP
    results["mAP"] = float(ap[valid_classes].mean())

    for iou_idx, iou_val in enumerate(iouv):
        results[f"mAP_{iou_val * 100:.0f}"] = float(ap[valid_classes, iou_idx].mean())

    # AP for each class
    for ci, name in enumerate(class_names):
        if ci in valid_classes:
            results[f"AP_{name}"] = float(ap[ci].mean())
        else:
            results[f"AP_{name}"] = 0.0

    # Compute global Precision / Recall (F1 best point)
    # Use the IoU=0.5 column
    tp0 = tp[:, 0].astype(np.float32)

    n_gt_all = len(target_cls)

    fpc = (1.0 - tp0).cumsum()
    tpc = tp0.cumsum()

    recall_curve = tpc / (n_gt_all + eps)
    precision_curve = tpc / (tpc + fpc + eps)

    f1 = 2 * precision_curve * recall_curve / (precision_curve + recall_curve + eps)
    best_idx = f1.argmax()

    results["precision"] = float(precision_curve[best_idx])
    results["recall"] = float(recall_curve[best_idx])

    return results


def compute_ap_single(recall: np.ndarray, precision: np.ndarray) -> float:
    """Compute AP given recall and precision curve."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # Precision monotonic decreasing "envelope"
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Find points where recall changes
    idx = np.where(mrec[1:] != mrec[:-1])[0]

    # Calculate area under curve
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)
