from typing import Sequence, Any, Mapping

import numpy as np
from ultralytics.utils.instance import Instances


def ensure_contiguous_output(arg: Any) -> Any:
    """
    Recursively make common data structures contiguous:
    - numpy arrays
    - torch tensors (if available)
    - Instances
    - dict / list / tuple / set
    """
    # 1 numpy array
    if isinstance(arg, np.ndarray):
        return np.ascontiguousarray(arg)

    # 2 torch.Tensor (optional)
    if "torch" in globals():
        import torch

        if isinstance(arg, torch.Tensor):
            return arg.contiguous()

    # 3 Instances
    if isinstance(arg, Instances):
        bboxes = np.ascontiguousarray(arg.bboxes)
        segments = (
            np.ascontiguousarray(arg.segments) if arg.segments is not None and len(arg.segments) else arg.segments
        )
        keypoints = np.ascontiguousarray(arg.keypoints) if arg.keypoints is not None else arg.keypoints
        return Instances(
            bboxes=bboxes,
            segments=segments,
            keypoints=keypoints,
            bbox_format=arg._bboxes.format,
            normalized=arg.normalized,
        )

    # 4 dict / Mapping
    if isinstance(arg, Mapping):
        return {k: ensure_contiguous_output(v) for k, v in arg.items()}

    # 5 Sequence (list, tuple, etc.), except str/bytes
    if isinstance(arg, Sequence) and not isinstance(arg, (str, bytes)):
        # keep orginal type
        return type(arg)(ensure_contiguous_output(x) for x in arg)

    # 6 set
    if isinstance(arg, set):
        return {ensure_contiguous_output(x) for x in arg}

    # 7 others
    return arg


def masks_to_overlap(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert instance masks to overlap mode.

    Args:
        masks (np.ndarray): Instance masks of shape (N, H, W).
    """
    if len(masks.shape) != 3:
        raise ValueError("Input masks should have shape (N, H, W).")

    N, H, W = masks.shape
    # Compute areas of each mask
    areas = masks.reshape(N, -1).sum(axis=1)
    # Sort instance by area (descending), like polygons2masks_overlap
    sorted_idx = np.argsort(-areas)
    masks_sorted = masks[sorted_idx]

    overlap_mask = np.zeros((H, W), dtype=np.int32 if N > 255 else np.uint8)

    for i in range(N):
        inst_mask = masks_sorted[i]
        inst_id = i + 1  # starting from 1 (0 = background)

        # overwrite but ensure max instance preserves priority
        overlap_mask = np.where(inst_mask == 1, inst_id, overlap_mask)

    return overlap_mask[None], sorted_idx


def overlap_to_masks(overlap_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert an overlap mask (H,W) back to multi-instance masks (N,H,W),
    and return sorted_idx such that:
        instance_id = i+1 corresponds to original index sorted_idx[i]
    """
    if overlap_mask.ndim != 2:
        raise ValueError("overlap_mask must have shape (H, W).")

    H, W = overlap_mask.shape

    # Find instance IDs (skip background 0)
    instance_ids = np.unique(overlap_mask)
    instance_ids = instance_ids[instance_ids > 0]

    if instance_ids.size == 0:
        return np.zeros((0, H, W), dtype=np.uint8), np.array([], dtype=np.int64)

    N = len(instance_ids)
    masks = np.zeros((N, H, W), dtype=np.uint8)

    # sorted idx Semantics:
    # instance IDs in overlap start from 1, and their order corresponds to the order in mask_sorted
    sorted_idx = np.zeros(N, dtype=np.int64)

    for i, inst_id in enumerate(instance_ids):
        # Construct N binary masks
        masks[i] = (overlap_mask == inst_id).astype(np.uint8)

        # The inst_id in overlap originally comes from masks_sorted[i]
        sorted_idx[i] = inst_id - 1

    return masks, sorted_idx
