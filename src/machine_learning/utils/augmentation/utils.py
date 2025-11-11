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
    # 1) numpy array
    if isinstance(arg, np.ndarray):
        return np.ascontiguousarray(arg)

    # 2) torch.Tensor (optional)
    if "torch" in globals():
        import torch

        if isinstance(arg, torch.Tensor):
            return arg.contiguous()

    # 3) Instances
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

    # 4) dict / Mapping
    if isinstance(arg, Mapping):
        return {k: ensure_contiguous_output(v) for k, v in arg.items()}

    # 5) Sequence (list, tuple, etc.), except str/bytes
    if isinstance(arg, Sequence) and not isinstance(arg, (str, bytes)):
        # keep orginal type
        return type(arg)(ensure_contiguous_output(x) for x in arg)

    # 6) set
    if isinstance(arg, set):
        return {ensure_contiguous_output(x) for x in arg}

    # 7) others
    return arg
