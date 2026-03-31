from typing import Callable, Any

import torch
import torch.nn as nn

from machine_learning.networks import BaseNet


def conv1x1(in_channels: int, out_channels: int, stride: int = 1):
    """1x1 convolution."""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_channels: int, out_channels: int, stride: int = 1, groups: int = 1, dilation: int = 1):
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


class BasicBlock(nn.Module):
    """The basic residual blocks for ResNet18 and ResNet34."""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        stride: int = 1,
        downsample: Callable = None,
        normalize_before: bool = False,
    ):
        super().__init__()
        self.stride = stride
        self.normalize_before = normalize_before
        self.out_channels = hidden_channels * self.expansion

        self.conv1 = conv3x3(in_channels, hidden_channels, stride)
        self.bn1 = nn.BatchNorm2d(in_channels) if self.normalize_before else nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.downsample = downsample

    def forward_pre(self, x: torch.Tensor) -> torch.Tensor:
        """ResNet v2 style: Pre-activation."""
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity

        return out

    def forward_post(self, x: torch.Tensor) -> torch.Tensor:
        """ResNet v1 style: Post-activation."""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize_before:
            return self.forward_pre(x)
        else:
            return self.forward_post(x)


class BottleNeck(nn.Module):
    """Bottleneck residual blocks for ResNet50, ResNet101 and ResNet152."""

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        stride: int = 1,
        downsample: Callable = None,
        normalize_before: bool = False,
    ):
        super().__init__()

        self.out_channels = hidden_channels * self.expansion
        self.normalize_before = normalize_before
        self.downsample = downsample
        self.stride = stride

        # 1x1 downsample -> 3x3 features extraction -> 1x1 upsample
        self.conv1 = conv1x1(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm2d(in_channels) if self.normalize_before else nn.BatchNorm2d(hidden_channels)
        self.conv2 = conv3x3(hidden_channels, hidden_channels, stride)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.conv3 = conv1x1(hidden_channels, hidden_channels * self.expansion)
        self.bn3 = nn.BatchNorm2d(hidden_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward_pre(self, x: torch.Tensor) -> torch.Tensor:
        """Resnet v2 style: Pre-activation."""
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out

    def forward_post(self, x: torch.Tensor) -> torch.Tensor:
        """Resnet v1 style: Post-activation."""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize_before:
            return self.forward_pre(x)
        else:
            return self.forward_post(x)


class ResNet(BaseNet):
    def __init__(
        self,
        block: type[BasicBlock] | type[BottleNeck],
        layers: list[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        normalize_before: bool = False,
        *args: tuple[Any, ...],
        **kwargs: dict[str, Any],
    ):
        super().__init__(args=args, kwargs=kwargs)
        self.in_channels = 64
        self.zero_init_residual = zero_init_residual
        self.normalize_before = normalize_before

        # Stem: 7x7 Conv + MaxPooling
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # The four stages correspond to FPN's res2, res3, res4, and res5
        self.layer1 = self._make_layer(block, layers[0], 64)
        self.layer2 = self._make_layer(block, layers[1], 128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], 256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], 512, stride=2)

        # Head: Global average pooling + fully connected classification headers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _initialize_weights(self) -> None:
        # Weight initialization
        super()._initialize_weights()

        # Zero-initialize the last BN of the residual block to enhance the stability of early training
        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleNeck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self, block: type[BasicBlock] | type[BottleNeck], block_num: int, hidden_channels: int, stride: int = 1
    ):
        downsample = None
        if stride != 1 or self.in_channels != hidden_channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, hidden_channels * block.expansion, stride),
                nn.BatchNorm2d(hidden_channels * block.expansion),
            )

        layers = []
        # The first block handles downsampling and changes in the number of channels
        layers.append(block(self.in_channels, hidden_channels, stride, downsample, self.normalize_before))
        self.in_channels = hidden_channels * block.expansion
        # The subsequent blocks keep the number of channels and spatial resolution unchanged
        for _ in range(1, block_num):
            layers.append(block(self.in_channels, hidden_channels, normalize_before=self.normalize_before))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor | dict[str, torch.Tensor]:
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Stages (C2 to C5)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # Dictionary output mode designed for object detection/segmentation/multi-exit
        if return_features:
            return {
                "res2": c2,  # stride 4
                "res3": c3,  # stride 8
                "res4": c4,  # stride 16
                "res5": c5,  # stride 32
            }

        # Standard classification output mode
        out = self.avgpool(c5)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


# ========================================================================
# Factory function: One-click generation of all versions of ResNet
# ========================================================================


def build_resnet(
    version: str = "resnet18",
    num_classes: int = 1000,
    zero_init_residual: bool = False,
    normalize_before: bool = False,
    **kwargs,
):
    """General ResNet build factory functions."""
    version = version.lower()

    if version == "resnet18":
        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, zero_init_residual, normalize_before, **kwargs)
    elif version == "resnet34":
        return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, zero_init_residual, normalize_before, **kwargs)
    elif version == "resnet50":
        return ResNet(BottleNeck, [3, 4, 6, 3], num_classes, zero_init_residual, normalize_before, **kwargs)
    elif version == "resnet101":
        return ResNet(BottleNeck, [3, 4, 23, 3], num_classes, zero_init_residual, normalize_before, **kwargs)
    elif version == "resnet152":
        return ResNet(BottleNeck, [3, 8, 36, 3], num_classes, zero_init_residual, normalize_before, **kwargs)
    else:
        raise ValueError(f"Unsupported ResNet version: {version}")
