from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from typing import Any, Callable, Self

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .layers import (
    NetworkBlock,
    PackedBatchNorm2d,
    PackedConv2d,
    PackedLinear,
    PackedNetworkBlock,
)

_PARAMETER_STORAGE_ALIGNMENT = 64


def _validate_depth(depth: int) -> int:
    if (depth - 4) % 6 != 0:
        raise ValueError(f"CIFAR WideResNet depth must satisfy depth = 6n + 4, got {depth}")
    blocks_per_stage = (depth - 4) // 6
    if blocks_per_stage < 1:
        raise ValueError(f"CIFAR WideResNet depth must be at least 10, got {depth}")
    return blocks_per_stage


def _align_offset(offset: int, alignment: int) -> int:
    if alignment < 1:
        raise ValueError(f"alignment must be >= 1, got {alignment}")
    remainder = offset % alignment
    return offset if remainder == 0 else offset + alignment - remainder


def _foreach_copy_(destinations: list[Tensor], sources: list[Tensor]) -> None:
    if destinations:
        torch._foreach_copy_(destinations, sources)


class MLP(nn.Module):
    """Standard single-model multi-layer perceptron.

    Activations are applied after every hidden layer, but not after the output
    layer.

    Shape:
        - Input: ``(B, in_features)``.
        - Output: ``(B, out_features)``.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Sequence[int],
        out_features: int,
        activation_layer: Callable[[], nn.Module] = nn.ReLU,
        bias: bool = True,
    ) -> None:
        """Initialize a single-model multi-layer perceptron.

        Args:
            in_features: Number of input features.
            hidden_features: Width of each hidden layer. An empty sequence
                creates a single linear projection.
            out_features: Number of output features.
            activation_layer: Factory that creates the activation module used
                after every hidden layer. Default: :class:`torch.nn.ReLU`.
            bias: If ``True``, each linear layer has a bias. Default: ``True``.
        """

        super().__init__()
        features = (in_features, *hidden_features, out_features)
        if any(width < 1 for width in features):
            raise ValueError(f"all feature widths must be >= 1, got {features}")

        self.in_features = in_features
        self.hidden_features = tuple(hidden_features)
        self.out_features = out_features
        self.layers = nn.ModuleList(
            [
                nn.Linear(input_width, output_width, bias=bias)
                for input_width, output_width in pairwise(features)
            ]
        )
        self.activations = nn.ModuleList(
            [activation_layer() for _ in self.hidden_features]
        )

    def forward(self, input: Tensor) -> Tensor:
        """Return MLP outputs for input shaped ``(B, in_features)``."""

        if input.ndim != 2:
            raise ValueError(f"MLP expects [B, F], got shape {tuple(input.shape)}")
        if input.shape[1] != self.in_features:
            raise ValueError(f"MLP expected F={self.in_features}, got F={input.shape[1]}")

        output = input
        for layer, activation in zip(self.layers[:-1], self.activations, strict=True):
            output = activation(layer(output))
        return self.layers[-1](output)


class PackedMLP(nn.Module):
    """Independent multi-layer perceptrons packed across local models.

    Each local model has independent weights and biases. Activations are
    applied after every hidden layer, but not after the output layer.

    Shape:
        - Input: ``(B, K, in_features)``.
        - Output: ``(B, K, out_features)``.
    """

    def __init__(
        self,
        num_models: int,
        in_features: int,
        hidden_features: Sequence[int],
        out_features: int,
        activation_layer: Callable[[], nn.Module] = nn.ReLU,
        bias: bool = True,
    ) -> None:
        """Initialize a packed multi-layer perceptron.

        Args:
            num_models: Number of independent local models, represented by the
                ``K`` input dimension.
            in_features: Number of features in each local model input.
            hidden_features: Width of each hidden layer. An empty sequence
                creates a single packed linear projection.
            out_features: Number of features in each local model output.
            activation_layer: Factory that creates the activation module used
                after every hidden layer. Default: :class:`torch.nn.ReLU`.
            bias: If ``True``, each packed linear layer has a bias.
                Default: ``True``.
        """

        super().__init__()
        features = (in_features, *hidden_features, out_features)
        if any(width < 1 for width in features):
            raise ValueError(f"all feature widths must be >= 1, got {features}")

        self.num_models = num_models
        self.in_features = in_features
        self.hidden_features = tuple(hidden_features)
        self.out_features = out_features
        self.layers = nn.ModuleList(
            [
                PackedLinear(
                    num_models=num_models,
                    in_features=input_width,
                    out_features=output_width,
                    bias=bias,
                )
                for input_width, output_width in pairwise(features)
            ]
        )
        self.activations = nn.ModuleList(
            [activation_layer() for _ in self.hidden_features]
        )

    def forward(self, input: Tensor) -> Tensor:
        """Return packed MLP outputs for input shaped ``(B, K, in_features)``."""

        output = input
        for layer, activation in zip(self.layers[:-1], self.activations, strict=True):
            output = activation(layer(output))
        return self.layers[-1](output)


class WideResNet(nn.Module):
    """Standard CIFAR-style Wide ResNet without packing or dropout.

    This is the non-packed reference model. It lazily provides a
    ``parameter_storage`` buffer shaped ``[1, D]`` so its parameters can use the
    same flattened storage layout as :class:`PackedWideResNet`.

    Shape:
        - Input: ``(B, C, H, W)``.
        - Output: ``(B, num_classes)``.

    Note:
        All trainable parameters own memory independently from
        ``parameter_storage``. Use the sync methods to explicitly copy between
        the model parameters and backing storage.
    """

    def __init__(
        self,
        depth: int,
        widen_factor: int,
        num_classes: int = 10,
        in_channels: int = 3,
    ) -> None:
        """Initialize a standard Wide ResNet.

        Args:
            depth: Network depth. Must satisfy ``depth = 6n + 4``; common
                values are ``16`` and ``28``.
            widen_factor: Width multiplier applied to the CIFAR Wide ResNet
                channels. Must be at least ``1``.
            num_classes: Number of classifier output classes. Default: ``10``.
            in_channels: Number of input image channels. Default: ``3``.
        """

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
        self._parameter_storage: Tensor | None
        self._parameter_names: tuple[str, ...]
        self._parameter_tensors: list[Tensor]
        self._storage_tensors: list[Tensor]

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
        self._init_parameter_storage()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _init_parameter_storage(self) -> None:
        named_parameters = tuple(self.named_parameters())
        self._parameter_names = tuple(name for name, _ in named_parameters)
        self._parameter_storage = None
        self._bind_parameter_storage_sync_tensors()
        self.register_load_state_dict_post_hook(self._clear_parameter_storage_after_load)

    def _bind_parameter_storage_sync_tensors(self) -> None:
        named_parameters = dict(self.named_parameters())
        self._parameter_tensors = []
        self._storage_tensors = []
        offset = 0
        for name in self._parameter_names:
            parameter = named_parameters[name]
            numel = parameter.numel()
            self._parameter_tensors.append(parameter.view(-1))
            if self._parameter_storage is not None:
                self._storage_tensors.append(
                    self._parameter_storage[0, offset : offset + numel]
                )
            offset += _align_offset(numel, _PARAMETER_STORAGE_ALIGNMENT)

    def _materialize_parameter_storage(self) -> None:
        if self._parameter_storage is not None:
            return
        first_parameter = next(self.parameters())
        self._parameter_storage = torch.zeros(
            1,
            self.parameter_storage_numel(),
            device=first_parameter.device,
            dtype=first_parameter.dtype,
        )
        self._bind_parameter_storage_sync_tensors()
        with torch.no_grad():
            _foreach_copy_(self._storage_tensors, self._parameter_tensors)

    def _clear_parameter_storage(self) -> None:
        self._parameter_storage = None
        self._bind_parameter_storage_sync_tensors()

    def _clear_parameter_storage_after_load(
        self,
        module: nn.Module,
        incompatible_keys: Any,
    ) -> None:
        del incompatible_keys
        assert module is self
        self._clear_parameter_storage()

    def _apply(self, fn: Callable[[Tensor], Tensor], recurse: bool = True) -> Self:
        result = super()._apply(fn, recurse)
        self._clear_parameter_storage()
        return result

    @property
    def parameter_storage(self) -> Tensor:
        """Materialize and return the non-persistent parameter backing storage."""

        self._materialize_parameter_storage()
        assert self._parameter_storage is not None
        return self._parameter_storage

    def parameter_storage_numel(self) -> int:
        """Return the aligned storage width ``D`` for ``parameter_storage``.

        Returns:
            The number of columns in ``parameter_storage``. This can be larger
            than the raw parameter count because each parameter segment is
            aligned and may include padding.

        Shape:
            ``parameter_storage`` has shape ``(1, D)``.
        """

        return sum(
            _align_offset(parameter.numel(), _PARAMETER_STORAGE_ALIGNMENT)
            for parameter in self.parameters()
        )

    def sync_storage_from_parameters_(self) -> Self:
        """Copy current model parameters into ``parameter_storage`` in-place.

        Returns:
            ``self``.

        Note:
            Call this after optimizer updates and before directly mixing or
            reading ``parameter_storage``. The method uses cached tensor views
            and ``torch._foreach_copy_`` to avoid rebuilding tensor lists on
            every call.
        """

        if self._parameter_storage is None:
            self._materialize_parameter_storage()
        else:
            with torch.no_grad():
                _foreach_copy_(self._storage_tensors, self._parameter_tensors)
        return self

    def sync_parameters_from_storage_(self) -> Self:
        """Copy ``parameter_storage`` values back into model parameters in-place.

        Returns:
            ``self``.

        Note:
            Call this after directly modifying ``parameter_storage`` and before
            the next forward pass.
        """

        self._materialize_parameter_storage()
        with torch.no_grad():
            _foreach_copy_(self._parameter_tensors, self._storage_tensors)
        return self

    def forward_features(self, input: Tensor) -> Tensor:
        """Return spatial features before global pooling and classification.

        Args:
            input: Image tensor.

        Shape:
            - Input: ``(B, C, H, W)``.
            - Output: ``(B, feature_channels, H / 4, W / 4)`` for the default
              CIFAR strides.

        Returns:
            The final convolutional feature map before global average pooling.
        """

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
        """Return classification logits.

        Args:
            input: Image tensor.

        Shape:
            - Input: ``(B, C, H, W)``.
            - Output: ``(B, num_classes)``.

        Returns:
            Classification logits.
        """

        out = self.forward_features(input)
        out = F.adaptive_avg_pool2d(out, output_size=1).flatten(1)
        return self.classifier(out)


class PackedWideResNet(nn.Module):
    """CIFAR-style Wide ResNet packed across independent local models.

    ``K`` local models are represented in one module. Parameters are isolated by
    grouped convolutions, packed linear weights, and BatchNorm over
    ``K * channels``.

    Shape:
        - Input: ``(B, K, C, H, W)``.
        - Output: ``(B, K, num_classes)``.

    Note:
        Internally, convolutional features are viewed as ``(B, K * C, H, W)``
        so grouped convolutions and BatchNorm2d can run without permuting axes.
        The ``parameter_storage`` buffer has shape ``(K, D)`` for decentralized
        parameter mixing. All trainable parameters own memory independently
        from this backing storage.
    """

    def __init__(
        self,
        depth: int,
        widen_factor: int,
        num_models: int,
        num_classes: int = 10,
        in_channels: int = 3,
    ) -> None:
        """Initialize a packed Wide ResNet.

        Args:
            depth: Network depth. Must satisfy ``depth = 6n + 4``; common
                values are ``16`` and ``28``.
            widen_factor: Width multiplier applied to the CIFAR Wide ResNet
                channels. Must be at least ``1``.
            num_models: Number of independent local models packed into this
                module. This is the ``K`` dimension of the input.
            num_classes: Number of classifier output classes. Default: ``10``.
            in_channels: Number of channels in each per-model input image.
                Default: ``3``.
        """

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
        self._parameter_storage: Tensor | None
        self._parameter_names: tuple[str, ...]
        self._parameter_tensors: list[Tensor]
        self._storage_tensors: list[Tensor]

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
        self._init_parameter_storage()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, PackedConv2d):
                nn.init.kaiming_normal_(
                    module.weight[: module.local_out_channels],
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias[: module.local_out_channels])
                module.broadcast_parameters_()
            elif isinstance(module, PackedBatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _init_parameter_storage(self) -> None:
        named_parameters = tuple(self.named_parameters())
        self._parameter_names = tuple(name for name, _ in named_parameters)
        self._parameter_storage = None
        self._bind_parameter_storage_sync_tensors()
        self.register_load_state_dict_post_hook(self._clear_parameter_storage_after_load)

    def _bind_parameter_storage_sync_tensors(self) -> None:
        named_parameters = dict(self.named_parameters())
        self._parameter_tensors = []
        self._storage_tensors = []
        offset = 0
        for name in self._parameter_names:
            parameter = named_parameters[name]
            local_numel = parameter.numel() // self.num_models
            self._parameter_tensors.append(parameter.view(self.num_models, local_numel))
            if self._parameter_storage is not None:
                self._storage_tensors.append(
                    self._parameter_storage[:, offset : offset + local_numel]
                )
            offset += _align_offset(local_numel, _PARAMETER_STORAGE_ALIGNMENT)

    def _materialize_parameter_storage(self) -> None:
        if self._parameter_storage is not None:
            return
        first_parameter = next(self.parameters())
        self._parameter_storage = torch.zeros(
            self.num_models,
            self.parameter_storage_numel(),
            device=first_parameter.device,
            dtype=first_parameter.dtype,
        )
        self._bind_parameter_storage_sync_tensors()
        with torch.no_grad():
            _foreach_copy_(self._storage_tensors, self._parameter_tensors)

    def _clear_parameter_storage(self) -> None:
        self._parameter_storage = None
        self._bind_parameter_storage_sync_tensors()

    def _clear_parameter_storage_after_load(
        self,
        module: nn.Module,
        incompatible_keys: Any,
    ) -> None:
        del incompatible_keys
        assert module is self
        self._clear_parameter_storage()

    def _apply(self, fn: Callable[[Tensor], Tensor], recurse: bool = True) -> Self:
        result = super()._apply(fn, recurse)
        self._clear_parameter_storage()
        return result

    @property
    def parameter_storage(self) -> Tensor:
        """Materialize and return the non-persistent parameter backing storage."""

        self._materialize_parameter_storage()
        assert self._parameter_storage is not None
        return self._parameter_storage

    def parameter_storage_numel(self) -> int:
        """Return the aligned per-model storage width ``D``.

        Returns:
            The number of columns in ``parameter_storage``. This can be larger
            than the raw per-model parameter count because each segment is
            aligned and may include padding.

        Shape:
            ``parameter_storage`` has shape ``(K, D)``.
        """

        return sum(
            _align_offset(
                parameter.numel() // self.num_models,
                _PARAMETER_STORAGE_ALIGNMENT,
            )
            for parameter in self.parameters()
        )

    def sync_storage_from_parameters_(self) -> Self:
        """Copy packed model parameters into ``parameter_storage`` in-place.

        Returns:
            ``self``.

        Note:
            Call this after optimizer updates and before applying decentralized
            mixing to ``parameter_storage``. The storage shape is ``(K, D)``, so
            a row-stochastic mixing matrix can be applied as
            ``parameter_storage.copy_(mixing @ parameter_storage)``.
        """

        if self._parameter_storage is None:
            self._materialize_parameter_storage()
        else:
            with torch.no_grad():
                _foreach_copy_(self._storage_tensors, self._parameter_tensors)
        return self

    def sync_parameters_from_storage_(self) -> Self:
        """Copy ``parameter_storage`` values back into packed parameters.

        Returns:
            ``self``.

        Note:
            Call this after modifying ``parameter_storage`` and before the next
            forward pass.
        """

        self._materialize_parameter_storage()
        with torch.no_grad():
            _foreach_copy_(self._parameter_tensors, self._storage_tensors)
        return self

    def average(self, model: WideResNet | None = None) -> WideResNet:
        """Return or update a ``WideResNet`` with the global parameter average.

        Args:
            model: Optional target model. If omitted, a matching ``WideResNet``
                is created on the same device and dtype. If provided, it is
                updated in-place and returned.

        Returns:
            A standard ``WideResNet`` whose parameters equal the average across
            the ``K`` rows of ``parameter_storage``.

        Note:
            Call ``sync_storage_from_parameters_()`` first if packed parameters
            may have changed since the last storage sync.
        """

        if model is None:
            model = WideResNet(
                depth=self.depth,
                widen_factor=self.widen_factor,
                num_classes=self.num_classes,
                in_channels=self.in_channels,
            )
            model.to(device=self.parameter_storage.device, dtype=self.parameter_storage.dtype)
        elif (
            model.depth != self.depth
            or model.widen_factor != self.widen_factor
            or model.num_classes != self.num_classes
            or model.in_channels != self.in_channels
        ):
            raise ValueError("target model must match packed depth, widen_factor, num_classes, and in_channels")
        if model.parameter_storage.shape[1] != self.parameter_storage.shape[1]:
            raise ValueError("target model parameter storage layout does not match packed model")

        with torch.no_grad():
            model.parameter_storage.copy_(self.parameter_storage.mean(dim=0, keepdim=True))
        model.sync_parameters_from_storage_()
        return model

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
        """Return packed spatial features before pooling and classification.

        Args:
            input: Packed image tensor where ``K`` equals ``num_models``.

        Shape:
            - Input: ``(B, K, C, H, W)``.
            - Output: ``(B, K * feature_channels, H / 4, W / 4)`` for the
              default CIFAR strides.

        Returns:
            The final packed convolutional feature map before global average
            pooling.
        """

        out = self._pack_input(input)
        out = self.stem(out)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = F.relu(self.bn(out), inplace=False)
        return out

    def forward(self, input: Tensor) -> Tensor:
        """Return packed classification logits.

        Args:
            input: Packed image tensor where ``K`` equals ``num_models``.

        Shape:
            - Input: ``(B, K, C, H, W)``.
            - Output: ``(B, K, num_classes)``.

        Returns:
            Classification logits. Each ``K`` slice is the prediction of one
            independent local model.
        """

        out = self.forward_features(input)
        out = F.adaptive_avg_pool2d(out, output_size=1).flatten(1)
        out = out.reshape(input.shape[0], self.num_models, self.feature_channels)
        return self.classifier(out)


def packed_wrn_28_10(
    num_models: int,
    num_classes: int = 10,
    in_channels: int = 3,
) -> PackedWideResNet:
    """Create a packed WRN-28-10.

    Args:
        num_models: Number of independent local models packed into one module.
        num_classes: Number of classifier output classes. Default: ``10``.
        in_channels: Number of channels in each per-model input image.
            Default: ``3``.

    Returns:
        A :class:`PackedWideResNet` with ``depth=28`` and ``widen_factor=10``.
    """

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
    """Create a packed WRN-16-8.

    Args:
        num_models: Number of independent local models packed into one module.
        num_classes: Number of classifier output classes. Default: ``10``.
        in_channels: Number of channels in each per-model input image.
            Default: ``3``.

    Returns:
        A :class:`PackedWideResNet` with ``depth=16`` and ``widen_factor=8``.
    """

    return PackedWideResNet(
        depth=16,
        widen_factor=8,
        num_models=num_models,
        num_classes=num_classes,
        in_channels=in_channels,
    )


def wrn_28_10(num_classes: int = 10, in_channels: int = 3) -> WideResNet:
    """Create a standard WRN-28-10.

    Args:
        num_classes: Number of classifier output classes. Default: ``10``.
        in_channels: Number of input image channels. Default: ``3``.

    Returns:
        A :class:`WideResNet` with ``depth=28`` and ``widen_factor=10``.
    """

    return WideResNet(
        depth=28,
        widen_factor=10,
        num_classes=num_classes,
        in_channels=in_channels,
    )


def wrn_16_8(num_classes: int = 10, in_channels: int = 3) -> WideResNet:
    """Create a standard WRN-16-8.

    Args:
        num_classes: Number of classifier output classes. Default: ``10``.
        in_channels: Number of input image channels. Default: ``3``.

    Returns:
        A :class:`WideResNet` with ``depth=16`` and ``widen_factor=8``.
    """

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

    Args:
        packed: Target packed model. Its ``num_models`` value must equal
            ``len(single_models)``.
        single_models: Source models. Each source may be a standard
            ``WideResNet`` or a ``PackedWideResNet`` with ``num_models=1``.

    Note:
        This utility is intended for validation and simulation setup where
        existing local model weights should be evaluated by one packed module.
        After the copy, ``packed.parameter_storage`` is synchronized and ready
        for averaging or decentralized mixing.
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
        packed.sync_storage_from_parameters_()
