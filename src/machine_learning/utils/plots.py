"""All the plot funs are in the format of np.ndarray."""

from typing import Literal

import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_imgs(
    imgs: list[np.ndarray],
    titles: list[str] | None = None,
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
        _plot_with_opencv(processed_images, titles)
    elif backend == "pillow":
        _plot_with_pillow(processed_images, titles)
    else:
        _plot_with_pyplot(processed_images, titles)


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

            else:
                # Multi-channel images, color mapping cannot be applied
                raise ValueError(f"The color mapping '{cmap}' can only be applied to single-channel images.")

        # Color inversion
        if reverse:
            img = 255 - img

    return img


def _plot_with_opencv(imgs: list[np.ndarray], titles: list[str] | None = None) -> None:
    for i, img in enumerate(imgs):
        window_name = titles[i] if titles is not None else f"Image: {i + 1}"
        cv2.imshow(window_name, img)

    cv2.waitKey(0)
    cv2.destroyWindow(window_name)


def _plot_with_pillow(imgs: list[np.ndarray], titles: list[str] | None = None) -> None:
    from PIL import Image

    for i, img in enumerate(imgs):
        # Make sure the value range is between 0-255
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        if img.ndim == 2:
            img = Image.fromarray(img, mode="L")
        else:
            img = Image.fromarray(img, mode="RGB")

        img.show(title=titles[i] if titles is not None else f"Image: {i + 1}")


def _plot_with_pyplot(imgs: list[np.ndarray], titles: list[str] | None = None) -> None:
    for i, img in enumerate(imgs):
        _ = plt.figure(i + 1, figsize=(6, 6))

        if img.ndim == 2:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(img)

        plt.axis("off")
        plt.title(titles[i] if titles is not None else f"Image: {i + 1}")
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

        if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
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
