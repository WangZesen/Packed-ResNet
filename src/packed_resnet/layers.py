from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class PackedConv2d(nn.Conv2d):
    """Grouped Conv2d where each group is one local model."""

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


class PackedBatchNorm2d(nn.BatchNorm2d):
    """BatchNorm2d over the packed channel dimension."""

    num_models: int
    local_num_features: int

    def __init__(self, num_models: int, num_features: int) -> None:
        if num_models < 1:
            raise ValueError(f"num_models must be >= 1, got {num_models}")
        self.num_models = num_models
        self.local_num_features = num_features
        super().__init__(num_models * num_features)


class PackedLinear(nn.Module):
    """Independent linear layers for each local model."""

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
        for weight in self.weight:
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
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
        output = torch.einsum("bkf,kof->bko", input, self.weight)
        if self.bias is not None:
            output = output + self.bias.unsqueeze(0)
        return output
