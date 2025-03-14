import torch  # noqa F:401
import torch.nn as nn
from typing import Sequence


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


def cal_conv_output_size(input_size: Sequence[int], conv_layer: nn.Module) -> Sequence[int]:
    if not isinstance(conv_layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        raise ValueError("conv_layer 必须是 nn.Conv1d、nn.Conv2d 或 nn.Conv3d 的实例")

    in_channels = input_size[1]
    if in_channels != conv_layer.in_channels:
        raise ValueError(f"输入通道数不匹配: 输入为 {in_channels}, 但卷积层要求 {conv_layer.in_channels}")

    # 计算核心逻辑
    def compute_dim(input_dim: int, kernel: int, padding: int, stride: int, dilation: int) -> int:
        return (input_dim + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1

    spatial_dims = input_size[2:]

    if isinstance(conv_layer, nn.Conv1d):
        if len(input_size) != 1:
            raise ValueError(f"Conv1d 输入需要 1 个空间维度，实际输入为 {spatial_dims}")
        k = conv_layer.kernel_size[0]
        p = conv_layer.padding[0]
        s = conv_layer.stride[0]
        d = conv_layer.dilation[0]
        new_length = compute_dim(spatial_dims[0], k, p, s, d)
        return input_size[0], conv_layer.out_channels, new_length

    elif isinstance(conv_layer, nn.Conv2d):
        if len(spatial_dims) != 2:
            raise ValueError(f"Conv2d 输入需要 2 个空间维度，实际输入为 {spatial_dims}")
        kh, kw = conv_layer.kernel_size
        ph, pw = conv_layer.padding
        sh, sw = conv_layer.stride
        dh, dw = conv_layer.dilation
        new_h = compute_dim(spatial_dims[0], kh, ph, sh, dh)
        new_w = compute_dim(spatial_dims[1], kw, pw, sw, dw)
        return (input_size[0], conv_layer.out_channels, new_h, new_w)

    else:
        if len(spatial_dims) != 3:
            raise ValueError(f"Conv3d 输入需要 3 个空间维度，实际输入为 {spatial_dims}")
        kd, kh, kw = conv_layer.kernel_size
        pd, ph, pw = conv_layer.padding
        sd, sh, sw = conv_layer.stride
        dd, dh, dw = conv_layer.dilation
        new_d = compute_dim(spatial_dims[0], kd, pd, sd, dd)
        new_h = compute_dim(spatial_dims[1], kh, ph, sh, dh)
        new_w = compute_dim(spatial_dims[2], kw, pw, sw, dw)
        return (input_size[0], conv_layer.out_channels, new_d, new_h, new_w)