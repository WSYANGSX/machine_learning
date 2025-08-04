from typing import Literal, Iterable, Sequence, Union

import time
import torch
import torchvision
import numpy as np
from copy import deepcopy

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


def to_abs_labels(bboxes: Union[torch.Tensor, np.ndarray], img_w: int, img_h: int) -> Union[torch.Tensor, np.ndarray]:
    new = deepcopy(bboxes)
    new[:, [0, 2]] *= img_w
    new[:, [1, 3]] *= img_h
    return new


def to_rel_labels(bboxes: Union[torch.Tensor, np.ndarray], img_w: int, img_h: int) -> Union[torch.Tensor, np.ndarray]:
    new = deepcopy(bboxes)
    new[:, [0, 2]] /= img_w
    new[:, [1, 3]] /= img_h
    return new


def yolo2voc(bboxes: Union[torch.Tensor, np.ndarray], img_w: int, img_h: int) -> np.ndarray:
    return to_abs_labels(xywh2xyxy(bboxes), img_w, img_h)


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
                    raise ValueError(f"pad_values must have length 1 or {img.shape[2]} for RGB images.")
                constant_values = [(v, v) for v in pad_values]  # (left, right) for each channel
            elif img.ndim == 2:  # gray
                if len(pad_values) != 1:
                    raise ValueError("pad_values must be a single int or a 1-tuple for grayscale images.")
                constant_values = pad_values[0]
        else:
            raise ValueError("pad_values must be an int or a tuple.")

    output = np.pad(img, pad_width, "constant", constant_values=constant_values)

    if output_type == "torch":
        output = torch.tensor(output, dtype=dtype)
        if output.ndim == 3:
            output = output.permute(2, 0, 1)

    return output


def rescale_boxes(
    boxes: Union[torch.Tensor, np.ndarray], img_size: int, org_img_w: int, org_img_h: int
) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert the bounding box coordinates output by the object detection model from the coordinate system of the padded
    square image back to the coordinate system of the original image.
    """
    # Calculate the increased pad and deal with the situation of expansion and contraction after the pad
    pad_x = max(org_img_h - org_img_w, 0) * (img_size / max(org_img_w, org_img_h))
    pad_y = max(org_img_w - org_img_h, 0) * (img_size / max(org_img_w, org_img_h))

    # the size after removing the pad
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x

    # remap the bounding box
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * org_img_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * org_img_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * org_img_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * org_img_h

    return boxes


def couple_bboxes_iou(
    bbox1: torch.Tensor,
    bbox2: torch.Tensor,
    bbox_format: Literal["pascal_voc", "coco"] = "pascal_voc",
    iou_type: Literal["default", "giou", "diou", "ciou"] = "default",
    eps: float = 1e-5,
    safe: bool = True,
) -> torch.Tensor:
    """Calculate the improved IoU to enhance numerical stability

    Args:
        bbox1 (torch.Tensor): bounding box1 with shape (N, 4).
        bbox2 (torch.Tensor): bounding box1 with shape (N, 4).
        bbox_format (Literal[&quot;pascal_voc&quot;, &quot;coco&quot;], optional): the format of bboxes, pascal_voc: [x_min, y_min, x_max, y_max] in absolute pixel coordinates, coco: [x_min, y_min, bbox_width, bbox_height] in absolute pixel coordinates..Defaults to "pascal_voc".
        iou_type (Literal[&quot;default&quot;, &quot;giou&quot;, &quot;diou&quot;, &quot;ciou&quot;], optional): the calculation type of iou. Defaults to "default".
        eps (float, optional): small positive numbers. Defaults to 1e-9.
        safe (bool, optional): safe mode or not. Defaults to True.

    Returns:
        torch.Tensor: result with shape (N, 4).
    """

    # 0. Input validation
    if safe:
        # Check whether the input is a finite value
        if torch.isnan(bbox1).any():
            raise ValueError("The input bounding-box bbox1 contains non-finite values (NaN).")
        elif torch.isinf(bbox1).any():
            raise ValueError("The input bounding-box bbox1 contains non-finite values (inf).")

        if torch.isnan(bbox2).any():
            raise ValueError("The input bounding-box bbox2 contains non-finite values (NaN).")
        elif torch.isinf(bbox2).any():
            raise ValueError("The input bounding-box bbox2 contains non-finite values (inf).")

        # Check the validity of the coordinates
        if bbox_format == "pascal_voc":
            if (bbox1[:, 2] < bbox1[:, 0]).any() or (bbox1[:, 3] < bbox1[:, 1]).any():
                raise ValueError("bbox1 contains invalid coordinates (x2 < x1 or y2 < y1).")
            if (bbox2[:, 2] < bbox2[:, 0]).any() or (bbox2[:, 3] < bbox2[:, 1]).any():
                raise ValueError("bbox2 contains invalid coordinates (x2 < x1 or y2 < y1).")
        else:
            if (bbox1[:, 2] < 0).any() or (bbox1[:, 3] < 0).any():
                raise ValueError("bbox1 contains invalid coordinates (w < 0 or h < 0).")
            if (bbox2[:, 2] < 0).any() or (bbox2[:, 3] < 0).any():
                raise ValueError("bbox2 contains invalid coordinates (w < 0 or h < 0).")

    # 1. Format conversion
    if bbox_format == "coco":
        bbox1 = xywh2xyxy(bbox1)
        bbox2 = xywh2xyxy(bbox2)

    # 2. Extract coordinates - Add dimension processing
    if bbox1.dim() == 1:
        bbox1 = bbox1.unsqueeze(0)
    if bbox2.dim() == 1:
        bbox2 = bbox2.unsqueeze(0)

    bb1_x1, bb1_y1, bb1_x2, bb1_y2 = bbox1[:, 0], bbox1[:, 1], bbox1[:, 2], bbox1[:, 3]
    bb2_x1, bb2_y1, bb2_x2, bb2_y2 = bbox2[:, 0], bbox2[:, 1], bbox2[:, 2], bbox2[:, 3]

    # 3. Calculate the intersection area (add protective clamp)
    inter_x1 = torch.max(bb1_x1, bb2_x1)
    inter_y1 = torch.max(bb1_y1, bb2_y1)
    inter_x2 = torch.min(bb1_x2, bb2_x2)
    inter_y2 = torch.min(bb1_y2, bb2_y2)

    inter_width = (inter_x2 - inter_x1).clamp(min=0)
    inter_height = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_width * inter_height

    # 4. Calculate the union region (add eps to all dimensions)
    bb1_width = (bb1_x2 - bb1_x1).clamp(min=eps)
    bb1_height = (bb1_y2 - bb1_y1).clamp(min=eps)
    bb2_width = (bb2_x2 - bb2_x1).clamp(min=eps)
    bb2_height = (bb2_y2 - bb2_y1).clamp(min=eps)

    union_area = (bb1_width * bb1_height) + (bb2_width * bb2_height) - inter_area + eps

    # 5.  Calculate basic IoU
    iou = inter_area / union_area

    # 6. Advanced IoU Variants
    if iou_type != "default":
        c_x1 = torch.min(bb1_x1, bb2_x1)
        c_y1 = torch.min(bb1_y1, bb2_y1)
        c_x2 = torch.max(bb1_x2, bb2_x2)
        c_y2 = torch.max(bb1_y2, bb2_y2)

        c_width = (c_x2 - c_x1).clamp(min=eps)
        c_height = (c_y2 - c_y1).clamp(min=eps)
        c_area = c_width * c_height

        if iou_type in ["diou", "ciou"]:
            bb1_cx = (bb1_x1 + bb1_x2) / 2
            bb1_cy = (bb1_y1 + bb1_y2) / 2
            bb2_cx = (bb2_x1 + bb2_x2) / 2
            bb2_cy = (bb2_y1 + bb2_y2) / 2

            center_dist_sq = (bb2_cx - bb1_cx).pow(2) + (bb2_cy - bb1_cy).pow(2)
            c_diagonal_sq = c_width.pow(2) + c_height.pow(2) + eps

            diou_term = center_dist_sq / c_diagonal_sq

            if iou_type == "ciou":
                arctan_diff = torch.atan(bb2_width / bb2_height.clamp(min=eps)) - torch.atan(
                    bb1_width / bb1_height.clamp(min=eps)
                )

                v = (4 / (torch.pi**2)) * arctan_diff.pow(2)

                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)

                return iou - diou_term - v * alpha
            else:
                return iou - diou_term
        else:
            return iou - (c_area - union_area) / c_area

    return iou


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def get_batch_statistics(detections: list[torch.Tensor], targets: torch.Tensor, iou_threshold: float) -> list:
    """Compute true positives, predicted scores and predicted labels per batch sample
    Source: https://https://github.com/eriklindernoren/PyTorch-YOLOv3

    Args:
        detections (torch.Tensor): list of detections after NMS of each imgs in the batch. detections (x1, y1, x2, y2, conf, cls)
        targets (torch.Tensor): real label data
        iou_threshold (float): iou threshold to filter true labels.

    Returns:
        list: statistics of the batch.
    """
    batch_metrics = []

    for i in range(len(detections)):
        detection = detections[i]
        pred_boxes = detection[:, :4]
        pred_scores = detection[:, 4]
        pred_clses = detection[:, -1]

        true_positives = torch.zeros(pred_boxes.shape[0], device=detection.device)

        annotations = targets[targets[:, 0] == i][:, 1:]  # annotations [cls, xyxy]
        target_cls = annotations[:, 0] if len(annotations) else []

        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for i, (pred_box, pred_cls) in enumerate(zip(pred_boxes, pred_clses)):
                # If all targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_cls not in target_cls:
                    continue

                # Filter target_boxes by pred_cls so that we only match against boxes of our own cls label
                filtered_target_indices, filtered_targets_boxes = zip(
                    *filter(lambda x: target_cls[x[0]] == pred_cls, enumerate(target_boxes))
                )

                # Find the best matching target for our predicted box
                iou, box_filtered_index = bbox_iou(pred_box.unsqueeze(0), torch.stack(filtered_targets_boxes)).max(0)

                # Remap the index in the list of filtered targets for that label to the index in the list with all targets.
                box_index = filtered_target_indices[box_filtered_index]

                # Check if the iou is above the min treshold and i
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[i] = 1
                    detected_boxes += [box_index]

        batch_metrics.append([true_positives, pred_scores, pred_clses])

    return batch_metrics


def average_precision_per_cls(tp, conf, pred_cls, target_cls):
    """Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(average_precision(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def average_precision(recall, precision):
    """Compute the average precision, given the recall and precision curves.
    Code originally from: https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def non_max_suppression(
    predictions: torch.Tensor,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    classes_filter: Iterable[int] = None,
    max_wh: int = 4096,
    max_det: int = 300,
    max_nms: int = 30000,
    time_limit: float = 1.0,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """Performs Non-Maximum Suppression (NMS) on inference results

    Args:
        prediction (torch.Tensor): predictions output from Darknet network. [B, h1*w1+h2*w2+h3*w3, 5+class_nums]
        conf_threshold (float): This threshold is used to filter out the prediction boxes whose confidence scores are
        lower than this threshold. Defaults to 0.25.
        iou_threshold (float): This threshold is used in the non-maximum suppression (NMS) process to determine which
        boxes overlap and should be merged or discarded. Defaults to 0.45.
        classes (list[str]): class filter. Defaults to None.
        max_wh (int): maximum box width and height. Defaults to 4096.
        max_det (int): maximum number of detections per image. Defaults to 300.
        max_nms (int): maximum number of boxes into torchvision.ops.nms(). Defaults to 30000.
        time_limit (float): seconds to quit after. Defaults to 1.0.

    Returns:
        torch.Tensor: detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    class_nums = predictions.shape[2] - 5  # number of classes
    multi_label = class_nums > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [torch.zeros((0, 6), device=device)] * predictions.shape[0]

    for i, prediction in enumerate(predictions):
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        prediction = prediction[prediction[..., 4] > conf_threshold]  # confidence

        # If none remain process next image
        if not prediction.shape[0]:
            continue

        # Compute conf
        prediction[:, 5:] *= prediction[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(prediction[:, :4])

        # Detections matrix nx6 (x1, y1, x2, y2, conf, cls)
        if multi_label:
            i, j = (prediction[:, 5:] > conf_threshold).nonzero(as_tuple=False).T  # i row indices, j col indices
            prediction = torch.cat((box[i], prediction[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = prediction[:, 5:].max(1, keepdim=True)
            prediction = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_threshold]

        # Filter by class
        if classes_filter is not None:
            prediction = prediction[
                (prediction[:, 5:6] == torch.tensor(classes_filter, device=prediction.device)).any(1)
            ]

        # Check shape
        num_boxes = prediction.shape[0]  # number of boxes
        if not num_boxes:  # no boxes
            continue
        elif num_boxes > max_nms:  # excess boxes
            # sort by confidence
            prediction = prediction[prediction[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = prediction[:, 5:6] * max_wh  # classes
        # boxes (offset by class), scores
        boxes, scores = prediction[:, :4] + c, prediction[:, 4]
        j = torchvision.ops.nms(boxes, scores, iou_threshold)  # NMS
        if j.shape[0] > max_det:  # limit detections
            j = j[:max_det]

        output[i] = prediction[j].cpu()

        if (time.time() - t) > time_limit:
            print(f"WARNING: NMS time limit {time_limit}s exceeded")
            break  # time limit exceeded

    return output
