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


def is_empty_array(arr: Union[torch.Tensor, np.ndarray]) -> bool:
    """Check if array is effectively empty."""
    return arr.size == 0 or (hasattr(arr, "shape") and 0 in arr.shape)


def img_tensor2np(img: torch.Tensor) -> np.ndarray:
    """Convert tensor format image to numpy arrays and adjust the format.

    Args:
        img (torch.Tensor): Input tensor of shape (C, H, W) or (H, W).

    Returns:
        Numpy array with shape (H, W, C) or (H, W) with values in range [0, 255] and dtype uint8.
    """
    # Input validation
    if not isinstance(img, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(img)}.")

    # Convert to numpy
    img = img.detach().cpu().numpy()

    if img.ndim == 3:
        # (C, H, W) -> (H, W, C)
        img = img.transpose(1, 2, 0)
        if img.shape[2] == 1:
            # Single channel: (H, W, 1) -> (H, W)
            img = img[:, :, 0]
    elif img.ndim == 2:
        # Single grayscale image: (H, W) -> keep as is
        pass
    else:
        raise ValueError(f"Unsupported tensor shape: {img.shape}.")

    # Normalize and convert to uint8
    if np.issubdtype(img.dtype, np.floating):
        if np.min(img) >= 0 and np.max(img) <= 1:
            img = (img * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    else:
        img = np.clip(img, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(img)


def img_np2tensor(img: np.ndarray) -> torch.Tensor:
    """Convert numpy array images to tensor format and adjust the format.

    Args:
        img (np.ndarray): Input array of shape (H, W, C) or (H, W)

    Returns:
        torch.Tensor: Tensor with shape (C, H, W) with values normalized to [0, 1] and dtype float32.
    """
    # Input validation
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(img)}.")

    # Make a copy to avoid modifying the original array
    img = img.copy()

    # Process value range and convert to float32
    if np.issubdtype(img.dtype, np.integer):
        # Integer types [0, 255] -> normalized float [0, 1]
        img = img.astype(np.float32) / 255.0
    elif np.issubdtype(img.dtype, np.floating):
        if np.min(img) < 0 or np.max(img) > 1:
            # Normalize to [0, 1] if not already normalized
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
        img = img.astype(np.float32)

    # Channel-last to channel-first conversion
    if img.ndim == 3:  # (H, W, C)
        # Single image: (H, W, C) -> (C, H, W)
        img = img.transpose(2, 0, 1)
    elif img.ndim == 2:
        # Single grayscale image: (H, W) -> (1, H, W)
        img = np.expand_dims(img, axis=0)

    else:
        raise ValueError(f"Unsupported array shape: {img.shape}.")

    # Convert to torch tensor
    return torch.from_numpy(img).contiguous()
