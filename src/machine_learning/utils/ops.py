from typing import Union

import torch
import numpy as np


def empty_like(x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Creates empty torch.Tensor or np.ndarray with same shape as input and float32 dtype."""
    return (
        torch.empty_like(x, dtype=torch.float32) if isinstance(x, torch.Tensor) else np.empty_like(x, dtype=np.float32)
    )


def zeros_like(x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Creates zeros torch.Tensor or np.ndarray with same shape as input and float32 dtype."""
    return (
        torch.zeros_like(x, dtype=torch.float32) if isinstance(x, torch.Tensor) else np.zeros_like(x, dtype=np.float32)
    )


def ones_like(x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Creates ones torch.Tensor or np.ndarray with same shape as input and float32 dtype."""
    return torch.ones_like(x, dtype=torch.float32) if isinstance(x, torch.Tensor) else np.ones_like(x, dtype=np.float32)
