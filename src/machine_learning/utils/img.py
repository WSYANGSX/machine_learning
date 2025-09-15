"""All the image drawing areas are in the format of np.ndarray."""

from typing import Literal

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt


def imgs_tensor2np(imgs: torch.Tensor, bs: bool | None = None) -> np.ndarray:
    """Convert tensor format images to numpy arrays and adjust the format.

    Args:
        imgs (torch.Tensor): Input tensor of shape (B, C, H, W) or (C, H, W) or (B, H, W) or (H, W)
        bs (bool | None): Whether the input tensor include a batch (multiple images). Default to None.

    Returns:
        Numpy array with shape (B, H, W, C) or (H, W, C) or (H, W) with values in range [0, 255] and dtype uint8.
    """
    # Input validation
    if not isinstance(imgs, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(imgs)}.")

    # Convert to numpy
    imgs = imgs.detach().cpu().numpy()

    # Determine if input has batch dimension
    if bs is None:
        # Auto-detect: if 4D or 3D with first dim > 4 (unlikely to be channels)
        bs = imgs.ndim == 4 or (imgs.ndim == 3 and imgs.shape[0] > 4)

    # Channel-first to channel-last conversion
    if imgs.ndim == 4:
        # (B, C, H, W) -> (B, H, W, C)
        imgs = imgs.transpose(0, 2, 3, 1)
    elif imgs.ndim == 3:
        if bs:
            # Batch of grayscale images: (B, H, W) -> (B, H, W, 1)
            imgs = imgs[..., np.newaxis]
        else:
            # Single image: (C, H, W) -> (H, W, C)
            imgs = imgs.transpose(1, 2, 0)
    elif imgs.ndim == 2:
        # Single grayscale image: (H, W) -> keep as is
        pass
    else:
        raise ValueError(f"Unsupported tensor shape: {imgs.shape}.")

    # Normalize and convert to uint8
    if np.issubdtype(imgs.dtype, np.floating):
        if np.min(imgs) >= 0 and np.max(imgs) <= 1:
            imgs = (imgs * 255).astype(np.uint8)
        else:
            imgs = np.clip(imgs, 0, 255).astype(np.uint8)
    else:
        imgs = np.clip(imgs, 0, 255).astype(np.uint8)

    return imgs


def imgs_np2tensor(imgs: np.ndarray, bs: bool | None = None) -> torch.Tensor:
    """Convert numpy array images to tensor format and adjust the format.

    Args:
        imgs (np.ndarray): Input array of shape (B, H, W, C) or (B, H, W) or (H, W, C) or (H, W)
        bs (bool | None): Whether the input tensor include a batch (multiple images). Default to None.

    Returns:
        torch.Tensor: Tensor with shape (B, C, H, W) or (C, H, W) or (H, W) with values normalized to [0, 1] and dtype float32.
    """
    # Input validation
    if not isinstance(imgs, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(imgs)}.")

    # Make a copy to avoid modifying the original array
    imgs = imgs.copy()

    # Determine if output should have batch dimension
    if bs is None:
        # Auto-detect: if 4D or 3D with first dim > 4 (unlikely to be height)
        bs = imgs.ndim == 4 or (imgs.ndim == 3 and imgs.shape[0] > 4)

    # Process value range and convert to float32
    if np.issubdtype(imgs.dtype, np.integer):
        # Integer types [0, 255] -> normalized float [0, 1]
        imgs = imgs.astype(np.float32) / 255.0
    elif np.issubdtype(imgs.dtype, np.floating):
        if np.min(imgs) < 0 or np.max(imgs) > 1:
            # Normalize to [0, 1] if not already normalized
            imgs = (imgs - np.min(imgs)) / (np.max(imgs) - np.min(imgs))
        imgs = imgs.astype(np.float32)

    # Channel-last to channel-first conversion
    if imgs.ndim == 4:
        # (B, H, W, C) -> (B, C, H, W)
        imgs = imgs.transpose(0, 3, 1, 2)
    elif imgs.ndim == 3:
        if bs:
            # batch images: (B, H, W) -> (B, 1, H, W)
            imgs = np.expand_dims(imgs, axis=1)
        else:
            # Single image: (H, W, C) -> (C, H, W)
            imgs = imgs.transpose(2, 0, 1)
    elif imgs.ndim == 2:
        # Single grayscale image: (H, W) -> (1, H, W)
        imgs = np.expand_dims(imgs, axis=0)

    else:
        raise ValueError(f"Unsupported array shape: {imgs.shape}.")

    # Convert to torch tensor
    return torch.from_numpy(imgs)


def plot_imgs(
    imgs: list[np.ndarray],
    cmap: str | None = "rgb",
    backend: Literal["pyplot", "opencv", "pillow"] = "pyplot",
) -> None:
    """Plot a single image.

    Args:
        imgs (np.ndarray): The images to plot.
        cmap: (str | None = None):cmap (str): Color map. Grayscale image: cmap='gray' or cmap='Greys', heatmap: cmap='hot', rainbow image: cmap='rainbow', blue-green gradient: cmap='viridis' (default), reversed color: Add r after any color mapping, such as cmap='viridis r'.
        backend (Literal[&quot;pyplot&quot;, &quot;opencv&quot;, &quot;pillow&quot;], optional): The plot backend. Defaults to "pyplot".
    """

    processed_images = [color_maps(img, cmap) for img in imgs]

    if backend == "opencv":
        _plot_with_opencv(processed_images)
    elif backend == "pillow":
        _plot_with_pillow(processed_images)
    else:
        _plot_with_pyplot(processed_images)


def color_maps(img: np.ndarray, cmap: str | None = "rgb") -> np.ndarray:
    """Preprocess the image, including value range normalization, channel order adjustment and color mode conversion.

    Args:
        img (np.ndarray): The input image.
        cmap: (str | None = None):cmap (str): Color map. Grayscale image: cmap='gray' or cmap='Greys', heatmap: cmap='hot', rainbow image: cmap='rainbow', blue-green gradient: cmap='viridis' (default), reversed color: Add r after any color mapping, such as cmap='viridis r'.

    Returns:
        np.ndarray: The preprocessed numpy array image.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(img)}.")

    if img.dtype == np.float32 or img.dtype == np.float64:
        if img.min() >= 0 and img.max() <= 1:
            img = (img * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    elif img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    if cmap is not None and img.ndim == 2:
        reverse = False
        if cmap.endswith(" r"):
            cmap = cmap[:-2]
            reverse = True

        # Apply color mapping
        if cmap.lower() in ["gray", "grey", "grays", "greys"]:
            # Convert to gray
            if img.ndim == 3 and img.shape[2] == 3:  # RGB to Gray
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            elif img.ndim == 3 and img.shape[2] == 4:  # RGBA to Gray
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        elif cmap.lower() == "rgb":
            # Convert to RGB
            if img.ndim == 2:  # Gray to RGB
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.ndim == 3 and img.shape[2] == 4:  # RGBA to RGB
                img = img[:, :, :3]
            elif img.ndim == 3 and img.shape[2] == 1:  # Single channel to RGB
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            # Apply color mapping (applicable to single-channel images)
            if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
                # Make sure it is a single channel
                if img.ndim == 3:
                    img = img[:, :, 0]

                if cmap.lower() == "hot":
                    img = cv2.applyColorMap(img, cv2.COLORMAP_HOT)
                elif cmap.lower() == "rainbow":
                    img = cv2.applyColorMap(img, cv2.COLORMAP_RAINBOW)
                elif cmap.lower() in ["viridis", "jet"]:
                    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                elif cmap.lower() == "cool":
                    img = cv2.applyColorMap(img, cv2.COLORMAP_COOL)
                elif cmap.lower() == "spring":
                    img = cv2.applyColorMap(img, cv2.COLORMAP_SPRING)
                elif cmap.lower() == "summer":
                    img = cv2.applyColorMap(img, cv2.COLORMAP_SUMMER)
                elif cmap.lower() == "autumn":
                    img = cv2.applyColorMap(img, cv2.COLORMAP_AUTUMN)
                elif cmap.lower() == "winter":
                    img = cv2.applyColorMap(img, cv2.COLORMAP_WINTER)
                elif cmap.lower() == "bone":
                    img = cv2.applyColorMap(img, cv2.COLORMAP_BONE)
                elif cmap.lower() == "ocean":
                    img = cv2.applyColorMap(img, cv2.COLORMAP_OCEAN)
                else:
                    # Viridis mapping is used by default
                    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)

                # Make sure the output is in RGB format (OpenCV defaults to BGR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                # Multi-channel images, color mapping cannot be applied
                raise ValueError(f"The color mapping '{cmap}' can only be applied to single-channel images.")

        # Color inversion
        if reverse:
            img = 255 - img

    # (H, W) -> (H, W, 1)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)

    return img


def _plot_with_opencv(imgs: list[np.ndarray]) -> None:
    for i, img in enumerate(imgs):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        window_name = f"Image: {i + 1}"
        cv2.imshow(window_name, img)

    cv2.waitKey(0)
    cv2.destroyWindow(window_name)


def _plot_with_pillow(imgs: list[np.ndarray]) -> None:
    from PIL import Image

    for i, img in enumerate(imgs):
        # Make sure the value range is between 0-255
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        if img.ndim == 2:
            img = Image.fromarray(img, mode="L")
        else:
            img = Image.fromarray(img, mode="RGB")

        img.show(title=f"Image: {i + 1}")


def _plot_with_pyplot(imgs: list[np.ndarray]) -> None:
    for i, img in enumerate(imgs):
        _ = plt.figure(i + 1, figsize=(6, 6))

        if img.ndim == 2:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(img)

        plt.axis("off")
        plt.title(f"Image: {i + 1}")
        plt.tight_layout()

    plt.show()


def plot_raw_and_recon_imgs(
    raw_imgs: list[np.ndarray],
    recon_imgs: list[np.ndarray],
    max_cols: int = 5,
    fig_size: tuple[int, int] = (12, 8),
    titles: list[str] = ["Raw images", "Recon images"],
) -> None:
    """Plot a comparison image between the raw images and the reconstructed images.

    Args:
        raw_imgs (list[np.ndarray]): The list of raw images.
        recon_imgs (list[np.ndarray]): The list of reconstructed images by generation algorithms.
        max_cols (int): The maximum number of images per line. Default to 5.
        figsize (tuple[int]): The size of the curtain.
        titles: (list[str]): The names of subplots.
    """
    _subplot_imgs(raw_imgs, max_cols=max_cols, title=titles[0], figure_size=fig_size)
    _subplot_imgs(recon_imgs, max_cols=max_cols, title=titles[1], figure_size=fig_size)

    plt.show()


def _subplot_imgs(imgs: list[np.ndarray], max_cols: int = 5, title: str | None = None, figure_size: tuple = (12, 8)):
    """
    Plot multiple images in a single figure.

    Args:
        imgs (list[np.ndarray]): The list of images.
        max_cols (int): The maximum number of images per line. Default to 5.
        title (str | None): The title of the figure. Default to None.
        figure_size (tuple[int, int]): The size of the figure.
    """
    n_imgs = len(imgs)
    n_rows = (n_imgs + max_cols - 1) // max_cols  # compute rows

    # Create figure and subfigures
    fig, axes = plt.subplots(n_rows, max_cols if n_rows > 1 else min(max_cols, n_imgs), figsize=figure_size)

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Set the general title
    fig.suptitle(title, fontsize=16)

    # Traverse all images and draw
    for i, img in enumerate(imgs):
        row = i // max_cols
        col = i % max_cols

        if len(img.shape) == 2:
            axes[row, col].imshow(img, cmap="gray")
        else:
            axes[row, col].imshow(img)

        # Set subfigure labels
        axes[row, col].set_xlabel(f"Image {i + 1}", fontsize=12)

        # Remove the coordinate axes
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])

    # Hide redundant subfigures
    for i in range(n_imgs, n_rows * max_cols):
        row = i // max_cols
        col = i % max_cols
        axes[row, col].axis("off")

    # Adjust the Layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for the main title

    return fig, axes
