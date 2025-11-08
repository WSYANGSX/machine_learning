from typing import Sequence
import numpy as np


def ensure_contiguous_output(arg: np.ndarray | Sequence[np.ndarray]) -> np.ndarray | list[np.ndarray]:
    """Ensure that numpy arrays are contiguous in memory.

    Args:
        arg (np.ndarray | Sequence[np.ndarray]): A numpy array or sequence of numpy arrays.

    Returns:
        np.ndarray | list[np.ndarray]: Contiguous array(s) with the same data.

    """
    if isinstance(arg, np.ndarray):
        arg = np.ascontiguousarray(arg)
    elif isinstance(arg, Sequence):
        arg = list(map(ensure_contiguous_output, arg))
    return arg
