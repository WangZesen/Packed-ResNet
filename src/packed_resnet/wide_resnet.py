from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .layers import PackedBatchNorm2d, PackedConv2d, PackedLinear


def _validate_depth(depth: int) -> int:
    if (depth - 4) % 6 != 0:
        raise ValueError(f"CIFAR WideResNet depth must satisfy depth = 6n + 4, got {depth}")
    blocks_per_stage = (depth - 4) // 6
    if blocks_per_stage < 1:
        raise ValueError(f"CIFAR WideResNet depth must be at least 10, got {depth}")
    return blocks_per_stage


class PackedBasicBlock(nn.Module):
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
        out = self.conv1(F.relu(self.bn1(input), inplace=False))
        out = self.conv2(F.relu(self.bn2(out), inplace=False))
        residual = input if self.shortcut is None else self.shortcut(input)
        return out + residual


class PackedNetworkBlock(nn.Sequential):
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
        out = self.conv1(F.relu(self.bn1(input), inplace=False))
        out = self.conv2(F.relu(self.bn2(out), inplace=False))
        residual = input if self.shortcut is None else self.shortcut(input)
        return out + residual


class NetworkBlock(nn.Sequential):
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


class WideResNet(nn.Module):
    """Standard CIFAR-style Wide ResNet without packing or dropout."""

    def __init__(
        self,
        depth: int,
        widen_factor: int,
        num_classes: int = 10,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        if widen_factor < 1:
            raise ValueError(f"widen_factor must be >= 1, got {widen_factor}")
        blocks_per_stage = _validate_depth(depth)

        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        self.depth = depth
        self.widen_factor = widen_factor
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feature_channels = channels[-1]

        self.stem = nn.Conv2d(
            in_channels,
            channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.stage1 = NetworkBlock(blocks_per_stage, channels[0], channels[1], stride=1)
        self.stage2 = NetworkBlock(blocks_per_stage, channels[1], channels[2], stride=2)
        self.stage3 = NetworkBlock(blocks_per_stage, channels[2], channels[3], stride=2)
        self.bn = nn.BatchNorm2d(channels[3])
        self.classifier = nn.Linear(channels[3], num_classes)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_features(self, input: Tensor) -> Tensor:
        if input.ndim != 4:
            raise ValueError(f"WideResNet expects [B, C, H, W], got {tuple(input.shape)}")
        if input.shape[1] != self.in_channels:
            raise ValueError(f"expected C={self.in_channels}, got C={input.shape[1]}")
        out = self.stem(input)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        return F.relu(self.bn(out), inplace=False)

    def forward(self, input: Tensor) -> Tensor:
        out = self.forward_features(input)
        out = F.adaptive_avg_pool2d(out, output_size=1).flatten(1)
        return self.classifier(out)


class PackedWideResNet(nn.Module):
    """CIFAR-style Wide ResNet packed across independent local models.

    The public input shape is ``[B, K, C, H, W]`` and output shape is
    ``[B, K, num_classes]``. Internally, convolutional features are viewed as
    ``[B, K * C, H, W]`` so grouped convolutions and BatchNorm2d can operate
    without permuting axes.
    """

    def __init__(
        self,
        depth: int,
        widen_factor: int,
        num_models: int,
        num_classes: int = 10,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        if num_models < 1:
            raise ValueError(f"num_models must be >= 1, got {num_models}")
        if widen_factor < 1:
            raise ValueError(f"widen_factor must be >= 1, got {widen_factor}")
        blocks_per_stage = _validate_depth(depth)

        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        self.depth = depth
        self.widen_factor = widen_factor
        self.num_models = num_models
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feature_channels = channels[-1]

        self.stem = PackedConv2d(
            num_models,
            in_channels,
            channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.stage1 = PackedNetworkBlock(
            blocks_per_stage, num_models, channels[0], channels[1], stride=1
        )
        self.stage2 = PackedNetworkBlock(
            blocks_per_stage, num_models, channels[1], channels[2], stride=2
        )
        self.stage3 = PackedNetworkBlock(
            blocks_per_stage, num_models, channels[2], channels[3], stride=2
        )
        self.bn = PackedBatchNorm2d(num_models, channels[3])
        self.classifier = PackedLinear(num_models, channels[3], num_classes)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, PackedConv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, PackedBatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _pack_input(self, input: Tensor) -> Tensor:
        if input.ndim != 5:
            raise ValueError(f"PackedWideResNet expects [B, K, C, H, W], got {tuple(input.shape)}")
        batch, num_models, channels, height, width = input.shape
        if num_models != self.num_models:
            raise ValueError(f"expected K={self.num_models}, got K={num_models}")
        if channels != self.in_channels:
            raise ValueError(f"expected C={self.in_channels}, got C={channels}")
        return input.reshape(batch, num_models * channels, height, width)

    def forward_features(self, input: Tensor) -> Tensor:
        out = self._pack_input(input)
        out = self.stem(out)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = F.relu(self.bn(out), inplace=False)
        return out

    def forward(self, input: Tensor) -> Tensor:
        out = self.forward_features(input)
        out = F.adaptive_avg_pool2d(out, output_size=1).flatten(1)
        out = out.reshape(input.shape[0], self.num_models, self.feature_channels)
        return self.classifier(out)


def packed_wrn_28_10(
    num_models: int,
    num_classes: int = 10,
    in_channels: int = 3,
) -> PackedWideResNet:
    return PackedWideResNet(
        depth=28,
        widen_factor=10,
        num_models=num_models,
        num_classes=num_classes,
        in_channels=in_channels,
    )


def packed_wrn_16_8(
    num_models: int,
    num_classes: int = 10,
    in_channels: int = 3,
) -> PackedWideResNet:
    return PackedWideResNet(
        depth=16,
        widen_factor=8,
        num_models=num_models,
        num_classes=num_classes,
        in_channels=in_channels,
    )


def wrn_28_10(num_classes: int = 10, in_channels: int = 3) -> WideResNet:
    return WideResNet(
        depth=28,
        widen_factor=10,
        num_classes=num_classes,
        in_channels=in_channels,
    )


def wrn_16_8(num_classes: int = 10, in_channels: int = 3) -> WideResNet:
    return WideResNet(
        depth=16,
        widen_factor=8,
        num_classes=num_classes,
        in_channels=in_channels,
    )


def copy_single_models_into_packed(
    packed: PackedWideResNet,
    single_models: Sequence[PackedWideResNet | WideResNet],
) -> None:
    """Copy K single WideResNets into one packed model.

    This utility is intended for validation and simulation setup where existing
    local model weights should be evaluated by one packed module.
    """

    if len(single_models) != packed.num_models:
        raise ValueError(f"expected {packed.num_models} single models, got {len(single_models)}")
    for model in single_models:
        if isinstance(model, PackedWideResNet) and model.num_models != 1:
            raise ValueError("packed source models must have num_models=1")
        if model.depth != packed.depth or model.widen_factor != packed.widen_factor:
            raise ValueError("all source models must match packed depth and widen_factor")
        if model.num_classes != packed.num_classes or model.in_channels != packed.in_channels:
            raise ValueError("all source models must match packed num_classes and in_channels")

    packed_modules = dict(packed.named_modules())
    source_module_dicts = [dict(model.named_modules()) for model in single_models]

    with torch.no_grad():
        for name, module in packed_modules.items():
            if isinstance(module, PackedConv2d):
                for model_idx, source_modules in enumerate(source_module_dicts):
                    source = source_modules[name]
                    assert isinstance(source, PackedConv2d | nn.Conv2d)
                    start = model_idx * module.local_out_channels
                    end = start + module.local_out_channels
                    module.weight[start:end].copy_(source.weight)
                    if module.bias is not None and source.bias is not None:
                        module.bias[start:end].copy_(source.bias)
            elif isinstance(module, PackedBatchNorm2d):
                assert isinstance(module.weight, Tensor)
                assert isinstance(module.bias, Tensor)
                assert isinstance(module.running_mean, Tensor)
                assert isinstance(module.running_var, Tensor)
                assert isinstance(module.num_batches_tracked, Tensor)
                for model_idx, source_modules in enumerate(source_module_dicts):
                    source = source_modules[name]
                    assert isinstance(source, PackedBatchNorm2d | nn.BatchNorm2d)
                    assert isinstance(source.weight, Tensor)
                    assert isinstance(source.bias, Tensor)
                    assert isinstance(source.running_mean, Tensor)
                    assert isinstance(source.running_var, Tensor)
                    start = model_idx * module.local_num_features
                    end = start + module.local_num_features
                    module.weight[start:end].copy_(source.weight)
                    module.bias[start:end].copy_(source.bias)
                    module.running_mean[start:end].copy_(source.running_mean)
                    module.running_var[start:end].copy_(source.running_var)
                first_source = source_module_dicts[0][name]
                assert isinstance(first_source, PackedBatchNorm2d | nn.BatchNorm2d)
                assert isinstance(first_source.num_batches_tracked, Tensor)
                module.num_batches_tracked.copy_(first_source.num_batches_tracked)
            elif isinstance(module, PackedLinear):
                for model_idx, source_modules in enumerate(source_module_dicts):
                    source = source_modules[name]
                    assert isinstance(source, PackedLinear | nn.Linear)
                    source_weight = source.weight[0] if isinstance(source, PackedLinear) else source.weight
                    module.weight[model_idx].copy_(source_weight)
                    if module.bias is not None and source.bias is not None:
                        source_bias = source.bias[0] if isinstance(source, PackedLinear) else source.bias
                        module.bias[model_idx].copy_(source_bias)
