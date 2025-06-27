from typing import Literal, Iterable, Sequence

import time
import torch
import torchvision
import numpy as np


def xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    new = x.new(x.shape)
    new[..., 0] = x[..., 0] - x[..., 2] / 2
    new[..., 1] = x[..., 1] - x[..., 3] / 2
    new[..., 2] = x[..., 0] + x[..., 2] / 2
    new[..., 3] = x[..., 1] + x[..., 3] / 2

    return new


def xywh2xyxy_np(x):
    new = np.zeros_like(x)
    new[..., 0] = x[..., 0] - x[..., 2] / 2
    new[..., 1] = x[..., 1] - x[..., 3] / 2
    new[..., 2] = x[..., 0] + x[..., 2] / 2
    new[..., 3] = x[..., 1] + x[..., 3] / 2

    return new


def xyxy2xywh(x: torch.Tensor) -> torch.Tensor:
    new = x.new(x.shape)
    new[..., 0] = (x[..., 0] + x[..., 2]) / 2
    new[..., 1] = (x[..., 1] + x[..., 3]) / 2
    new[..., 2] = x[..., 2] - x[..., 0]
    new[..., 3] = x[..., 3] - x[..., 1]

    return new


def xyxy2xywh_np(x: torch.Tensor) -> torch.Tensor:
    new = np.zeros_like(x)
    new[..., 0] = (x[..., 0] + x[..., 2]) / 2
    new[..., 1] = (x[..., 1] + x[..., 3]) / 2
    new[..., 2] = x[..., 2] - x[..., 0]
    new[..., 3] = x[..., 3] - x[..., 1]

    return new


def to_absolute_labels(img: torch.Tensor | np.ndarray, bboxes: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    img, bboxes = img, bboxes
    h, w, _ = img.shape
    bboxes[:, [0, 2]] *= w
    bboxes[:, [1, 3]] *= h
    return bboxes


def to_relative_labels(img: torch.Tensor | np.ndarray, bboxes: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    img, bboxes = img, bboxes
    h, w, _ = img.shape
    bboxes[:, [0, 2]] /= w
    bboxes[:, [1, 3]] /= h
    return bboxes


def yolo2voc(img: torch.Tensor | np.ndarray, bboxes: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    return to_absolute_labels(img, xywh2xyxy_np(bboxes))


def pad_to_square(img: np.ndarray, pad_values: int | Sequence[tuple[int]]):
    h, w = img.shape[0], img.shape[1]
    dim_diff = np.abs(h - w)
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2

    # 处理不同维度的图像
    if img.ndim == 2:  # 灰度图 (H, W)
        pad_width = [(pad1, pad2), (0, 0)] if h <= w else [(0, 0), (pad1, pad2)]
    elif img.ndim == 3:  # 彩色图 (H, W, C)
        pad_width = [(pad1, pad2), (0, 0), (0, 0)] if h <= w else [(0, 0), (pad1, pad2), (0, 0)]
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    # 确保pad_values兼容
    if isinstance(pad_values, int):
        constant_values = pad_values
    else:
        constant_values = pad_values

    return np.pad(img, pad_width, "constant", constant_values=constant_values)


def rescale_padded_boxes(boxes: torch.Tensor, cur_img_size: int, original_img_shape: tuple[int]) -> torch.Tensor:
    """
    将目标检测模型输出的边界框坐标从padding后的正方形图像尺寸转换回原始图像尺寸,
    [example](/home/yangxf/WorkSpace/machine_learning/docs/pictures/01.jpg)
    """
    _, orig_h, orig_w = original_img_shape

    # 计算增加的pad, 应对pad后放缩的情况
    pad_x = max(orig_h - orig_w, 0) * (cur_img_size / max(original_img_shape))
    pad_y = max(orig_w - orig_h, 0) * (cur_img_size / max(original_img_shape))

    # 移除pad后的尺寸
    unpad_h = cur_img_size - pad_y
    unpad_w = cur_img_size - pad_x

    # 重新映射边界框
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h

    return boxes


def bbox_iou(
    bbox1: torch.Tensor,
    bbox2: torch.Tensor,
    bbox_format: Literal["pascal_voc", "coco"] = "pascal_voc",
    iou_type: Literal["default", "giou", "diou", "ciou"] = "default",
    eps: float = 1e-9,
    safe: bool = True,
) -> torch.Tensor:
    """Calculate the improved IoU to enhance numerical stability

    Args:
        bbox1 (torch.Tensor): bounding box1 with shape (N, 4) or (1, 4).
        bbox2 (torch.Tensor): bounding box1 with shape (N, 4) or (1, 4).
        bbox_format (Literal[&quot;pascal_voc&quot;, &quot;coco&quot;], optional): the format of bboxes, pascal_voc: [x_min, y_min, x_max, y_max] in absolute pixel coordinates, coco: [x_min, y_min, bbox_width, bbox_height] in absolute pixel coordinates..Defaults to "pascal_voc".
        iou_type (Literal[&quot;default&quot;, &quot;giou&quot;, &quot;diou&quot;, &quot;ciou&quot;], optional): the calculation type of iou. Defaults to "default".
        eps (float, optional): small positive numbers. Defaults to 1e-9.
        safe (bool, optional): safe mode or not. Defaults to True.

    Returns:
        torch.Tensor: result with shape (N, 4) or (1, 4).
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
                iou, box_filtered_index = bbox_iou(
                    pred_box.unsqueeze(0), torch.stack(filtered_targets_boxes), bbox_format="coco"
                ).max(0)

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
