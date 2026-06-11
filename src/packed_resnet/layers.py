from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class PackedConv2d(nn.Conv2d):
    """Apply independent convolutions for packed local models.

    ``num_models`` independent convolutions are represented as one grouped
    :class:`torch.nn.Conv2d`. Inputs and outputs use contiguous channel groups
    for each local model.

    Args:
        num_models: Number of independent local models.
        in_channels: Input channels per local model.
        out_channels: Output channels per local model.
        kernel_size: Convolution kernel size.
        stride: Convolution stride. Default: ``1``.
        padding: Implicit zero padding. Default: ``0``.
        bias: If ``True``, include an independent bias for each model.

    Shape:
        - Input: ``(B, num_models * in_channels, H, W)``.
        - Output: ``(B, num_models * out_channels, H_out, W_out)``.

    Raises:
        ValueError: If ``num_models`` is less than one.
    """

    num_models: int
    local_in_channels: int
    local_out_channels: int

    def __init__(
        self,
        num_models: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        bias: bool = False,
    ) -> None:
        if num_models < 1:
            raise ValueError(f"num_models must be >= 1, got {num_models}")
        self.num_models = num_models
        self.local_in_channels = in_channels
        self.local_out_channels = out_channels
        super().__init__(
            in_channels=num_models * in_channels,
            out_channels=num_models * out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=num_models,
        )

    def reset_parameters(self) -> None:
        super().reset_parameters()
        self.broadcast_parameters_()

    def broadcast_parameters_(self) -> None:
        """Copy model-0 parameters to every other packed local model."""

        if self.num_models == 1:
            return
        with torch.no_grad():
            weight = self.weight.view(
                self.num_models,
                self.local_out_channels,
                self.local_in_channels,
                *self.kernel_size,
            )
            weight[1:].copy_(weight[0].unsqueeze(0).expand_as(weight[1:]))
            if self.bias is not None:
                bias = self.bias.view(self.num_models, self.local_out_channels)
                bias[1:].copy_(bias[0].unsqueeze(0).expand_as(bias[1:]))


class PackedBatchNorm2d(nn.BatchNorm2d):
    """Apply BatchNorm2d to packed local-model channel groups.

    Each local model owns separate affine parameters and running statistics.

    Args:
        num_models: Number of independent local models.
        num_features: Feature channels per local model.

    Shape:
        - Input and output: ``(B, num_models * num_features, H, W)``.

    Raises:
        ValueError: If ``num_models`` is less than one.
    """

    num_models: int
    local_num_features: int

    def __init__(self, num_models: int, num_features: int) -> None:
        if num_models < 1:
            raise ValueError(f"num_models must be >= 1, got {num_models}")
        self.num_models = num_models
        self.local_num_features = num_features
        super().__init__(num_models * num_features)


class PackedLinear(nn.Module):
    """Apply independent linear projections for packed local models.

    Args:
        num_models: Number of independent local models.
        in_features: Input features per local model.
        out_features: Output features per local model.
        bias: If ``True``, include an independent bias for each model.

    Shape:
        - Input: ``(B, num_models, in_features)``.
        - Output: ``(B, num_models, out_features)``.

    Note:
        Parameters for model 0 are initialized once and copied to every other
        local model.

    Raises:
        ValueError: If ``num_models`` is less than one.
    """

    def __init__(
        self,
        num_models: int,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if num_models < 1:
            raise ValueError(f"num_models must be >= 1, got {num_models}")
        self.num_models = num_models
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(num_models, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(num_models, out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight[0], a=math.sqrt(5))
        if self.num_models > 1:
            with torch.no_grad():
                self.weight[1:].copy_(
                    self.weight[0].unsqueeze(0).expand_as(self.weight[1:])
                )
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias[0], -bound, bound)
            if self.num_models > 1:
                with torch.no_grad():
                    self.bias[1:].copy_(
                        self.bias[0].unsqueeze(0).expand_as(self.bias[1:])
                    )

    def forward(self, input: Tensor) -> Tensor:
        """Apply each local model's linear projection.

        Args:
            input: Packed feature tensor shaped ``(B, K, in_features)``.

        Returns:
            Packed output tensor shaped ``(B, K, out_features)``.
        """

        if input.ndim != 3:
            raise ValueError(f"PackedLinear expects [B, K, F], got shape {tuple(input.shape)}")
        if input.shape[1] != self.num_models:
            raise ValueError(
                f"PackedLinear expected K={self.num_models}, got K={input.shape[1]}"
            )
        if input.shape[2] != self.in_features:
            raise ValueError(
                f"PackedLinear expected F={self.in_features}, got F={input.shape[2]}"
            )
        output = torch.bmm(input.transpose(0, 1), self.weight.transpose(1, 2))
        if self.bias is not None:
            output = output + self.bias.unsqueeze(1)
        return output.transpose(0, 1)


class PackedBasicBlock(nn.Module):
    """Pre-activation residual block for packed local models."""

    def __init__(
        self,
        num_models: int,
        in_channels: int,
        out_channels: int,
        stride: int,
    ) -> None:
        super().__init__()
        self.bn1 = PackedBatchNorm2d(num_models, in_channels)
        self.conv1 = PackedConv2d(
            num_models,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = PackedBatchNorm2d(num_models, out_channels)
        self.conv2 = PackedConv2d(
            num_models,
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.shortcut: PackedConv2d | None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = PackedConv2d(
                num_models,
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
        else:
            self.shortcut = None

    def forward(self, input: Tensor) -> Tensor:
        preactivated = F.relu(self.bn1(input), inplace=False)
        out = self.conv1(preactivated)
        out = self.conv2(F.relu(self.bn2(out), inplace=False))
        residual = input if self.shortcut is None else self.shortcut(preactivated)
        return out + residual


class PackedNetworkBlock(nn.Sequential):
    """Sequence of packed residual blocks."""

    def __init__(
        self,
        num_layers: int,
        num_models: int,
        in_channels: int,
        out_channels: int,
        stride: int,
    ) -> None:
        layers: list[PackedBasicBlock] = []
        for layer_idx in range(num_layers):
            layers.append(
                PackedBasicBlock(
                    num_models=num_models,
                    in_channels=in_channels if layer_idx == 0 else out_channels,
                    out_channels=out_channels,
                    stride=stride if layer_idx == 0 else 1,
                )
            )
        super().__init__(*layers)


class BasicBlock(nn.Module):
    """Pre-activation residual block for a standard Wide ResNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
    ) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.shortcut: nn.Conv2d | None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
        else:
            self.shortcut = None

    def forward(self, input: Tensor) -> Tensor:
        preactivated = F.relu(self.bn1(input), inplace=False)
        out = self.conv1(preactivated)
        out = self.conv2(F.relu(self.bn2(out), inplace=False))
        residual = input if self.shortcut is None else self.shortcut(preactivated)
        return out + residual


class NetworkBlock(nn.Sequential):
    """Sequence of standard residual blocks."""

    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        out_channels: int,
        stride: int,
    ) -> None:
        layers: list[BasicBlock] = []
        for layer_idx in range(num_layers):
            layers.append(
                BasicBlock(
                    in_channels=in_channels if layer_idx == 0 else out_channels,
                    out_channels=out_channels,
                    stride=stride if layer_idx == 0 else 1,
                )
            )
        super().__init__(*layers)
