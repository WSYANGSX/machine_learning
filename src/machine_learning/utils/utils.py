import math
import torch
import torch.nn as nn
import numpy as np

from typing import Sequence

import matplotlib.pyplot as plt


def print_dict(input_dict: dict, indent: int = 0) -> None:
    indent = indent

    for key, val in input_dict.items():
        print("\t" * indent, end="")
        if isinstance(val, dict):
            indent += 1
            print(key, ":")
            print_dict(val, indent)
            indent = 0
        else:
            print(key, ":", end="")
            print("\t", val)


def cal_conv_output_size(input_size: Sequence[int], conv_layer: nn.Module) -> tuple[int]:
    if not isinstance(conv_layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        raise ValueError("conv_layer 必须是 pytorch 卷积层实例")

    in_channels = input_size[0]
    if in_channels != conv_layer.in_channels:
        raise ValueError(f"输入通道数不匹配: 输入为 {in_channels}, 但卷积层要求 {conv_layer.in_channels}")

    # 计算核心逻辑
    def compute_dim(input_dim: int, kernel: int, padding: int, stride: int, dilation: int) -> int:
        return (input_dim + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1

    spatial_dims = input_size[1:]

    if isinstance(conv_layer, nn.Conv1d):
        if len(spatial_dims) != 1:
            raise ValueError(f"Conv1d 输入数据需要 1 个维度，实际输入为 {spatial_dims}")
        k = conv_layer.kernel_size[0]
        p = conv_layer.padding[0]
        s = conv_layer.stride[0]
        d = conv_layer.dilation[0]
        new_length = compute_dim(spatial_dims[0], k, p, s, d)
        return conv_layer.out_channels, new_length

    elif isinstance(conv_layer, nn.Conv2d):
        if len(spatial_dims) != 2:
            raise ValueError(f"Conv2d 输入数据需要 2 个维度，实际输入为 {spatial_dims}")
        kh, kw = conv_layer.kernel_size
        ph, pw = conv_layer.padding
        sh, sw = conv_layer.stride
        dh, dw = conv_layer.dilation
        new_h = compute_dim(spatial_dims[0], kh, ph, sh, dh)
        new_w = compute_dim(spatial_dims[1], kw, pw, sw, dw)
        return conv_layer.out_channels, new_h, new_w

    else:
        if len(spatial_dims) != 3:
            raise ValueError(f"Conv3d 输入数据需要 3 个维度，实际输入为 {spatial_dims}")
        kd, kh, kw = conv_layer.kernel_size
        pd, ph, pw = conv_layer.padding
        sd, sh, sw = conv_layer.stride
        dd, dh, dw = conv_layer.dilation
        new_d = compute_dim(spatial_dims[0], kd, pd, sd, dd)
        new_h = compute_dim(spatial_dims[1], kh, ph, sh, dh)
        new_w = compute_dim(spatial_dims[2], kw, pw, sw, dw)
        return conv_layer.out_channels, new_d, new_h, new_w


def cal_convtrans_output_size(input_size: Sequence[int], convtrans_layer: nn.Module) -> tuple[int]:
    if not isinstance(convtrans_layer, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        raise ValueError("convtrans_layer 必须是 PyTorch 的转置卷积层实例")

    in_channels = input_size[0]
    spatial_dims = input_size[1:]

    if in_channels != convtrans_layer.in_channels:
        raise ValueError(f"输入通道数不匹配: 输入为 {in_channels}, 但转置卷积层要求 {convtrans_layer.in_channels}")

    def compute_dim(input_dim: int, kernel: int, padding: int, stride: int, dilation: int, output_padding: int) -> int:
        return (input_dim - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1

    if isinstance(convtrans_layer, nn.ConvTranspose1d):
        if len(spatial_dims) != 1:
            raise ValueError(f"ConvTranspose1d 输入数据需要 1 个维度，实际输入为 {spatial_dims}")
        k = convtrans_layer.kernel_size[0]
        p = convtrans_layer.padding[0]
        s = convtrans_layer.stride[0]
        d = convtrans_layer.dilation[0]
        op = convtrans_layer.output_padding[0]
        new_length = compute_dim(spatial_dims[0], k, p, s, d, op)

        return convtrans_layer.out_channels, new_length

    elif isinstance(convtrans_layer, nn.ConvTranspose2d):
        if len(spatial_dims) != 2:
            raise ValueError(f"ConvTranspose2d 输入数据需要 2 个维度，实际输入为 {spatial_dims}")
        kh, kw = convtrans_layer.kernel_size[0], convtrans_layer.kernel_size[1]
        ph, pw = convtrans_layer.padding[0], convtrans_layer.padding[1]
        sh, sw = convtrans_layer.stride[0], convtrans_layer.stride[1]
        dh, dw = convtrans_layer.dilation[0], convtrans_layer.dilation[1]
        oph, opw = convtrans_layer.output_padding[0], convtrans_layer.output_padding[1]
        new_h = compute_dim(spatial_dims[0], kh, ph, sh, dh, oph)
        new_w = compute_dim(spatial_dims[1], kw, pw, sw, dw, opw)

        return convtrans_layer.out_channels, new_h, new_w

    elif isinstance(convtrans_layer, nn.ConvTranspose3d):
        if len(spatial_dims) != 3:
            raise ValueError(f"ConvTranspose2d 输入数据需要 3 个维度，实际输入为 {spatial_dims}")
        kd, kh, kw = convtrans_layer.kernel_size[0], convtrans_layer.kernel_size[1], convtrans_layer.kernel_size[2]
        pd, ph, pw = convtrans_layer.padding[0], convtrans_layer.padding[1], convtrans_layer.padding[2]
        sd, sh, sw = convtrans_layer.stride[0], convtrans_layer.stride[1], convtrans_layer.stride[2]
        dd, dh, dw = convtrans_layer.dilation[0], convtrans_layer.dilation[1], convtrans_layer.dilation[2]
        opd, oph, opw = (
            convtrans_layer.output_padding[0],
            convtrans_layer.output_padding[1],
            convtrans_layer.output_padding[2],
        )
        new_d = compute_dim(spatial_dims[0], kd, pd, sd, dd, opd)
        new_h = compute_dim(spatial_dims[1], kh, ph, sh, dh, oph)
        new_w = compute_dim(spatial_dims[2], kw, pw, sw, dw, opw)

        return convtrans_layer.out_channels, new_d, new_h, new_w


def cal_pooling_output_size(input_size: Sequence[int], pooling_layer: nn.Module) -> tuple[int]:
    if not isinstance(
        pooling_layer, (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d, nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)
    ):
        raise ValueError("pooling_layer 必须是 PyTorch 的池化层实例")

    # 提取channels
    channels = input_size[0]
    spatial_dims = input_size[1:]

    # 计算核心逻辑
    def compute_dim(
        input_dim: int, kernel: int, padding: int, stride: int, dilation: int = 1, ceil_mode: bool = False
    ) -> int:
        numerator = input_dim + 2 * padding - dilation * (kernel - 1)
        if ceil_mode:
            return (numerator + stride - 1) // stride
        else:
            # 向下取整逻辑
            return (numerator - 1) // stride + 1

    # 根据池化类型提取参数
    if isinstance(pooling_layer, (nn.AvgPool1d, nn.MaxPool1d)):
        expected_dims = 1
        kernel = pooling_layer.kernel_size
        padding = pooling_layer.padding
        stride = pooling_layer.stride
        ceil_mode = pooling_layer.ceil_mode if hasattr(pooling_layer, "ceil_mode") else False
        dilation = getattr(pooling_layer, "dilation", (1,))[0] if isinstance(pooling_layer, nn.MaxPool1d) else 1

    elif isinstance(pooling_layer, (nn.AvgPool2d, nn.MaxPool2d)):
        expected_dims = 2
        kernel = pooling_layer.kernel_size
        padding = pooling_layer.padding
        stride = pooling_layer.stride
        ceil_mode = pooling_layer.ceil_mode if hasattr(pooling_layer, "ceil_mode") else False
        dilation = getattr(pooling_layer, "dilation", (1, 1)) if isinstance(pooling_layer, nn.MaxPool2d) else (1, 1)

    elif isinstance(pooling_layer, (nn.AvgPool3d, nn.MaxPool3d)):
        expected_dims = 3
        kernel = pooling_layer.kernel_size
        padding = pooling_layer.padding
        stride = pooling_layer.stride
        ceil_mode = pooling_layer.ceil_mode if hasattr(pooling_layer, "ceil_mode") else False
        dilation = (
            getattr(pooling_layer, "dilation", (1, 1, 1)) if isinstance(pooling_layer, nn.MaxPool3d) else (1, 1, 1)
        )

    else:
        raise ValueError("不支持的池化类型")

    # 校验空间维度
    if len(spatial_dims) != expected_dims:
        raise ValueError(f"{type(pooling_layer).__name__} 需要 {expected_dims} 个空间维度，输入为 {len(spatial_dims)}")

    # 计算每个空间维度的新尺寸
    new_dims = []
    for i in range(expected_dims):
        new_dim = compute_dim(
            input_dim=spatial_dims[i],
            kernel=kernel[i] if isinstance(kernel, tuple) else kernel,
            padding=padding[i] if isinstance(padding, tuple) else padding,
            stride=stride[i] if isinstance(stride, tuple) else stride,
            dilation=dilation[i] if isinstance(dilation, tuple) else dilation,
            ceil_mode=ceil_mode,
        )
        new_dims.append(new_dim)

    return channels, *new_dims


def plot_raw_recon_figures(raw_figures: torch.Tensor | np.ndarray, recon_figures: torch.Tensor | np.ndarray):
    plt.figure(figsize=(10, 4))
    num_figures = len(raw_figures)

    for i in range(len(raw_figures)):
        # 原始图像
        ax = plt.subplot(2, num_figures, i + 1)
        plt.imshow(raw_figures[i].cpu().squeeze(), cmap="gray")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # 重构图像
        ax = plt.subplot(2, num_figures, i + 1 + num_figures)
        plt.imshow(recon_figures[i].cpu().squeeze(), cmap="gray")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()


def plot_figures(figures: torch.Tensor | np.ndarray, cmap: str):
    if isinstance(figures, torch.Tensor):
        figures = figures.cpu().numpy()

    if figures.ndim == 3:
        figures = figures[None, ...]

    plt.figure(figsize=(10, 10))

    figures_num = len(figures)
    col_num = 4
    row_num = math.ceil(figures_num / 4)

    for row in range(row_num):
        for col in range(col_num):
            # 计算当前图像的索引
            index = row * col_num + col
            if index < figures_num:
                ax = plt.subplot(row_num, col_num, index + 1)
                plt.imshow(figures[index].squeeze(), cmap=cmap)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.show()
