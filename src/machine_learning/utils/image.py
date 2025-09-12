from typing import Sequence, Mapping, Literal, overload

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from machine_learning.types.aliases import FilePath


BOX_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (255, 255, 255)  # White


def add_bbox(
    img: np.ndarray | torch.Tensor,
    bbox: np.ndarray | torch.Tensor,
    class_name: str,
    color: tuple[int] = BOX_COLOR,
    thickness: int = 2,
) -> np.ndarray | torch.Tensor:
    """向图像中添加单个边界框（修复顶部标签显示问题）

    Args:
        img (np.ndarray | torch.Tensor): 要添加边界框的图像
        bbox (np.ndarray | torch.Tensor): 边界框参数, Pascal VOC格式(x_min, y_min, x_max, y_max)
        class_name (str): 边界框中物体的类别名称
        color (tuple[int], optional): 边界框颜色，默认红色
        thickness (int, optional): 边界框的粗细, 默认2

    Returns:
        np.ndarray | torch.Tensor: 添加边界框后的图像
    """
    # 确保处理的是numpy数组（如果是Tensor则转换）
    if isinstance(img, torch.Tensor):
        img_np = img.cpu().numpy().transpose(1, 2, 0) if img.dim() == 3 else img.cpu().numpy()
        return_tensor = True
    else:
        img_np = img.copy()
        return_tensor = False

    # 获取图像尺寸
    img_height, img_width = img_np.shape[:2]

    # 转换坐标为整数
    x_min, y_min, x_max, y_max = map(int, bbox)

    # 绘制边界框
    cv2.rectangle(img_np, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    # 获取文本尺寸
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    label_height = int(1.3 * text_height)

    # 智能调整标签位置（防止超出图像边界）
    if y_min - label_height < 0:  # 顶部空间不足
        # 将标签放在边界框内部底部
        label_y_top = y_min
        label_y_bottom = label_y_top + label_height
        text_y = label_y_top + label_height - int(0.3 * text_height)
    else:  # 正常情况：标签放在边界框上方
        label_y_top = y_min - label_height
        label_y_bottom = y_min
        text_y = y_min - int(0.3 * text_height)

    # 确保标签不超出图像底部
    if label_y_bottom > img_height:
        label_y_top = max(0, img_height - label_height)
        label_y_bottom = img_height
        text_y = label_y_bottom - int(0.3 * text_height)

    # 绘制标签背景和文本
    cv2.rectangle(img_np, (x_min, label_y_top), (x_min + text_width, label_y_bottom), BOX_COLOR, -1)
    cv2.putText(
        img_np,
        text=class_name,
        org=(x_min, text_y),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )

    # 如果需要，转换回Tensor
    if return_tensor:
        img = torch.from_numpy(img_np.transpose(2, 0, 1)) if img_np.ndim == 3 else torch.from_numpy(img_np)
    else:
        img = img_np

    return img


def visualize_img_with_bboxes(
    img: np.ndarray,
    bboxes: np.ndarray,
    category_ids: Sequence[int],
    category_id_to_name: Sequence[str] | Mapping[int, str],
    cmap: str | None = None,
) -> None:
    """显示添加边界框后的图像

    Args:
        image (np.ndarray): 要添加边界框的图像.
        bboxes (np.ndarray): 边界框, 以pascal_voc格式输入(x_min, y_min, x_max, y_max).
        category_ids (Sequence[int]): 边界框中物体的类别编号序列.
        category_id_to_name (Sequence[str]): 边界框中物体的类别编号对应的名称序列.
    """
    img = img.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = add_bbox(img, bbox, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis("off")
    plt.imshow(img, cmap)
    plt.show()


@overload
def show_image(
    img: torch.Tensor | np.ndarray,
    color_mode: Literal["rgb", "gray"],
    backend: Literal["pyplot", "opencv", "pillow"] = "pyplot",
): ...


@overload
def show_image(
    imgs: torch.Tensor | np.ndarray,
    color_mode: Literal["rgb", "gray"],
    backend: Literal["pyplot", "opencv", "pillow"] = "pyplot",
): ...


def show_image(
    img: torch.Tensor | np.ndarray | None = None,
    imgs: Sequence[torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray | None = None,
    color_mode: Literal["rgb", "gray"] = "rgb",
    backend: Literal["pyplot", "opencv", "pillow"] = "pyplot",
):
    """
    显示单个或多个图像

    参数说明:
    figure - 单个图像 (Tensor 或 ndarray)
    figures - 多个图像列表、元组等或者numpy、Tensor数组
    color_mode - 颜色模式: "rgb" 或 "gray"
    backend - 显示后端: "pyplot" | "opencv" | "pillow"

    注意: figure 和 figures 参数互斥，只能使用其中一个
    """

    if img is not None and imgs is not None:
        raise ValueError("You can not input img and imgs simultaneously.")

    if img is None and imgs is None:
        raise ValueError("You must provide one of img or imgs parameter.")

    processed_images = []

    if img is not None:  # 单张图片
        processed_images.append(image_preprocess(img, color_mode))

    if imgs is not None:  # 多张图片
        imgs = [*imgs]
        for img in imgs:
            processed_images.append(image_preprocess(img, color_mode))

    # 显示图像
    if backend == "opencv":
        _show_with_opencv(processed_images)
    elif backend == "pillow":
        _show_with_pillow(processed_images)
    else:
        _show_with_pyplot(processed_images)


def image_preprocess(img: torch.Tensor | np.ndarray, color_mode: Literal["gray", "rgb"]) -> np.ndarray:
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    # 确保是numpy数组
    if not isinstance(img, np.ndarray):
        raise TypeError(f"不支持的图像类型: {type(img)}")

    # 处理值范围 (自动检测并归一化到[0, 255])
    if img.dtype == np.float32 or img.dtype == np.float64:
        if img.min() >= 0 and img.max() <= 1:
            img = (img * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    elif img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)

    # 处理通道顺序
    if img.ndim == 3 and img.shape[0] in [1, 3]:  # (C,H,W)
        img = img.transpose(1, 2, 0).squeeze()  # 转为HWC,并压缩掉维度为1的轴

    # 处理颜色模式
    if color_mode == "gray":
        if img.ndim == 3:  # 彩色转灰度
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        if img.ndim == 2:  # 灰度转rgb
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 4:  # RGBA转RGB
            img = img[:, :, :3]


def _show_with_opencv(imgs: list[np.ndarray]) -> None:
    for i, img in enumerate(imgs):
        # OpenCV需要BGR格式
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        window_name = f"Image {i + 1}"
        cv2.imshow(window_name, img)

    cv2.waitKey(0)
    cv2.destroyWindow(window_name)


def _show_with_pillow(imgs: list[np.ndarray]) -> None:
    from PIL import Image

    for i, img in enumerate(imgs):
        # 确保值范围在0-255
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        # 灰度图需要特殊处理
        if img.ndim == 2:
            img = Image.fromarray(img, mode="L")
        else:
            img = Image.fromarray(img, mode="RGB")

        img.show(title=f"Image {i + 1}")


def _show_with_pyplot(imgs: list[np.ndarray]) -> None:
    n = len(imgs)

    # 创建子图布局
    if n == 1:
        fig, ax = plt.subplots(figsize=(6, 6))
        axs = [ax]
    else:
        fig, axs = plt.subplots(1, n, figsize=(n * 4, 4))

    # 显示每个图像
    for i, (ax, img) in enumerate(zip(axs, imgs)):
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"Image {i + 1}")

    plt.tight_layout()
    plt.show()


def show_raw_and_recon_images(
    raw_imgs: torch.Tensor | np.ndarray,
    recon_imgs: torch.Tensor | np.ndarray,
    max_per_row: int = 5,
    figsize: tuple = (12, 8),
    titles: Sequence[str] = ("Raw Images", "Reconstructed Images"),
) -> None:
    """
    显示原始图像和重构图像的对比图

    参数:
    raw_imgs - 原始图像 (Tensor 或 ndarray), 形状为 (N, C, H, W) 或 (N, H, W, C)
    recon_imgs - 重构图像 (Tensor 或 ndarray), 形状应与 raw_imgs 相同
    max_per_row - 每行最多显示图像数量 (默认为5)
    figsize - 整个图像的大小 (宽度, 高度)
    titles - 上下两行的标题 (原始图像标题, 重构图像标题)
    """
    # 转换输入为 numpy 数组
    raw_np = _to_numpy(raw_imgs)
    recon_np = _to_numpy(recon_imgs)

    # 确保图像数量一致
    n_images = raw_np.shape[0]
    if n_images != recon_np.shape[0]:
        raise ValueError(f"原始图像数量({n_images})与重构图像数量({recon_np.shape[0]})不一致")

    # 计算需要多少行 (每行最多 max_per_row 个图像)
    n_rows = (n_images + max_per_row - 1) // max_per_row

    # 创建图像布局 (2行：原始图像 + 重构图像)
    fig, axes = plt.subplots(
        2, max_per_row if n_images <= max_per_row else min(n_images, max_per_row * n_rows), figsize=figsize
    )

    # 如果只有一张图像，axes 不是二维数组，需要调整
    if n_images == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    # 绘制原始图像 (上排)
    _plot_image_row(axes[0], raw_np, n_images, max_per_row, n_rows, titles[0])

    # 绘制重构图像 (下排)
    _plot_image_row(axes[1], recon_np, n_images, max_per_row, n_rows, titles[1])

    plt.tight_layout()
    plt.show()


def _to_numpy(imgs: torch.Tensor | np.ndarray) -> np.ndarray:
    """将输入转换为 numpy 数组并调整格式"""
    # 转换 Tensor 到 numpy
    if isinstance(imgs, torch.Tensor):
        imgs = imgs.detach().cpu().numpy()

    # 确保是 numpy 数组
    if not isinstance(imgs, np.ndarray):
        raise TypeError(f"不支持的图像类型: {type(imgs)}")

    # 调整维度顺序 (N, C, H, W) -> (N, H, W, C)
    if imgs.ndim == 4 and imgs.shape[1] in [1, 3]:
        imgs = imgs.transpose(0, 2, 3, 1)

    # 处理值范围 (浮点数归一化到0-255)
    if imgs.dtype == np.float32 or imgs.dtype == np.float64:
        if imgs.min() >= 0 and imgs.max() <= 1:
            imgs = (imgs * 255).astype(np.uint8)
        else:
            imgs = np.clip(imgs, 0, 255).astype(np.uint8)

    return imgs


def _plot_image_row(ax_row, images, n_images, max_per_row, n_rows, title):
    """在单行轴对象上绘制图像"""
    # 设置行标题
    ax_row[0].set_ylabel(title, fontsize=12, rotation=0, labelpad=40, va="center")

    # 遍历该行所有轴
    for j, ax in enumerate(ax_row):
        idx = j  # 计算当前图像索引

        # 如果图像数量不足，隐藏多余的轴
        if idx >= n_images:
            ax.axis("off")
            continue

        # 显示图像
        if images[idx].ndim == 2:  # 灰度图
            ax.imshow(images[idx], cmap="gray")
        else:  # 彩色图
            ax.imshow(images[idx])

        # 设置子标题 (只显示第一列)
        if j == 0:
            ax.set_title(f"Image {idx + 1}", fontsize=9)

        ax.axis("off")


def read_img_to_normalize_tensor(img_path: FilePath, mean: list[float], std: list[float]) -> torch.Tensor:
    import cv2
    from torchvision.transforms import Compose, ToTensor, Normalize

    image = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)
    tfs = Compose([ToTensor(), Normalize(mean=mean, std=std)])

    return tfs(image)


def resize(image: torch.Tensor, size: int) -> torch.Tensor:
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image
