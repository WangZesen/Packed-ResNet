from __future__ import annotations

import pytest
import torch

from packed_resnet import (
    PackedBatchNorm2d,
    PackedConv2d,
    PackedLinear,
    PackedWideResNet,
    WideResNet,
    copy_single_models_into_packed,
    packed_wrn_28_10,
    wrn_28_10,
)
from packed_resnet.layers import BasicBlock, PackedBasicBlock


def _num_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _storage_ptr(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


def _channels_last_input(batch_size: int, channels: int) -> torch.Tensor:
    return torch.randn(batch_size, channels, 32, 32).contiguous(
        memory_format=torch.channels_last
    )


def _packed_input(batch_size: int, num_models: int, channels: int = 3) -> torch.Tensor:
    return _channels_last_input(batch_size, num_models * channels)


def _local_tensor_view(tensor: torch.Tensor, num_models: int) -> torch.Tensor:
    local_numel = tensor.numel() // num_models
    if tensor.is_contiguous():
        return tensor.view(num_models, local_numel)
    assert tensor.ndim == 4
    assert tensor.is_contiguous(memory_format=torch.channels_last)
    return tensor.as_strided((num_models, local_numel), (local_numel, 1))


def _assert_local_parameters_identical(model: PackedWideResNet) -> None:
    for parameter in model.parameters():
        local_parameters = _local_tensor_view(parameter, model.num_models)
        torch.testing.assert_close(
            local_parameters,
            local_parameters[0].unsqueeze(0).expand_as(local_parameters),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize(
    ("block", "input"),
    [
        (BasicBlock(in_channels=2, out_channels=4, stride=2), torch.randn(2, 2, 8, 8)),
        (
            PackedBasicBlock(num_models=3, in_channels=2, out_channels=4, stride=2),
            torch.randn(2, 3 * 2, 8, 8),
        ),
    ],
)
def test_projection_shortcut_uses_preactivated_input(
    block: BasicBlock | PackedBasicBlock,
    input: torch.Tensor,
) -> None:
    block.eval()
    assert block.shortcut is not None

    with torch.no_grad():
        block.conv1.weight.zero_()
        block.conv2.weight.zero_()
        preactivated = torch.relu(block.bn1(input))
        expected = block.shortcut(preactivated)
        raw_shortcut = block.shortcut(input)

    output = block(input)

    torch.testing.assert_close(output, expected)
    assert not torch.allclose(output, raw_shortcut)


@pytest.mark.parametrize(
    ("block", "input"),
    [
        (BasicBlock(in_channels=2, out_channels=2, stride=1), torch.randn(2, 2, 8, 8)),
        (
            PackedBasicBlock(num_models=3, in_channels=2, out_channels=2, stride=1),
            torch.randn(2, 3 * 2, 8, 8),
        ),
    ],
)
def test_identity_shortcut_uses_raw_input(
    block: BasicBlock | PackedBasicBlock,
    input: torch.Tensor,
) -> None:
    block.eval()
    assert block.shortcut is None

    with torch.no_grad():
        block.conv1.weight.zero_()
        block.conv2.weight.zero_()

    torch.testing.assert_close(block(input), input)


def test_forward_shape() -> None:
    model = PackedWideResNet(depth=10, widen_factor=1, num_models=3, num_classes=7)
    x = _packed_input(2, 3)

    logits = model(x)

    assert logits.shape == (2, 3, 7)


def test_packed_wide_resnet_initializes_all_local_models_identically() -> None:
    model = PackedWideResNet(depth=10, widen_factor=1, num_models=4, num_classes=7)

    _assert_local_parameters_identical(model)


def test_wide_resnet_forward_shape() -> None:
    model = WideResNet(depth=10, widen_factor=1, num_classes=7)
    x = _channels_last_input(2, 3)

    logits = model(x)

    assert logits.shape == (2, 7)


def test_parameter_count_scales_exactly_with_num_models() -> None:
    single = PackedWideResNet(depth=10, widen_factor=2, num_models=1, num_classes=5)
    normal = WideResNet(depth=10, widen_factor=2, num_classes=5)
    packed = PackedWideResNet(depth=10, widen_factor=2, num_models=4, num_classes=5)

    assert _num_parameters(packed) == 4 * _num_parameters(single)
    assert _num_parameters(packed) == 4 * _num_parameters(normal)


def test_parameter_storage_is_lazy_non_persistent_and_aligned() -> None:
    packed = PackedWideResNet(depth=10, widen_factor=1, num_models=3, num_classes=5)
    normal = WideResNet(depth=10, widen_factor=1, num_classes=5)

    assert packed._parameter_storage is None
    assert normal._parameter_storage is None
    assert "_parameter_storage" not in packed.state_dict()
    assert "_parameter_storage" not in normal.state_dict()

    assert packed.parameter_storage.shape == (3, packed.parameter_storage_numel())
    assert normal.parameter_storage.shape == (1, normal.parameter_storage_numel())
    assert packed.parameter_storage.shape[1] == normal.parameter_storage.shape[1]

    for model, num_models in ((packed, packed.num_models), (normal, 1)):
        assert model._parameter_names == tuple(name for name, _ in model.named_parameters())
        assert len(model._parameter_names) == len(model._storage_tensors)
        offset = 0
        for parameter in model.parameters():
            local_numel = parameter.numel() // num_models
            padded_numel = (local_numel + 63) // 64 * 64
            padding = model.parameter_storage[:, offset + local_numel : offset + padded_numel]
            assert torch.count_nonzero(padding) == 0
            offset += padded_numel
        assert offset == model.parameter_storage.shape[1]


def test_loading_state_dict_invalidates_materialized_parameter_storage() -> None:
    source = PackedWideResNet(depth=10, widen_factor=1, num_models=2, num_classes=5)
    target = PackedWideResNet(depth=10, widen_factor=1, num_models=2, num_classes=5)
    source_parent = torch.nn.ModuleDict({"model": source})
    target_parent = torch.nn.ModuleDict({"model": target})
    target.parameter_storage.zero_()

    target_parent.load_state_dict(source_parent.state_dict())

    assert target._parameter_storage is None
    target_storage = target.parameter_storage
    source_storage = source.parameter_storage
    torch.testing.assert_close(target_storage, source_storage)


def test_parameter_storage_is_excluded_from_torch_export() -> None:
    model = PackedWideResNet(depth=10, widen_factor=1, num_models=2, num_classes=5).eval()
    model.parameter_storage

    exported = torch.export.export(model, (_packed_input(1, 2),))

    assert all("parameter_storage" not in name for name in exported.state_dict)
    assert all("parameter_storage" not in name for name in exported.constants)


def test_parameters_are_independent_from_parameter_storage() -> None:
    packed = PackedWideResNet(depth=10, widen_factor=1, num_models=2, num_classes=5)
    normal = WideResNet(depth=10, widen_factor=1, num_classes=5)

    for model in (packed, normal):
        storage_ptr = _storage_ptr(model.parameter_storage)
        assert all(_storage_ptr(parameter) != storage_ptr for parameter in model.parameters())

        converted = model.to(dtype=torch.float64)
        converted_storage_ptr = _storage_ptr(converted.parameter_storage)
        assert converted.parameter_storage.dtype == torch.float64
        assert all(
            _storage_ptr(parameter) != converted_storage_ptr for parameter in converted.parameters()
        )


def test_parameter_storage_changes_only_through_explicit_sync() -> None:
    torch.manual_seed(0)
    models = [
        PackedWideResNet(depth=10, widen_factor=1, num_models=2, num_classes=5),
        WideResNet(depth=10, widen_factor=1, num_classes=5),
    ]

    for model in models:
        storage_before = model.parameter_storage.detach().clone()
        parameter_before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        changed_name, changed_parameter = next(iter(model.named_parameters()))

        with torch.no_grad():
            changed_parameter.add_(1)
        torch.testing.assert_close(model.parameter_storage, storage_before)

        model.sync_storage_from_parameters_()
        assert not torch.equal(model.parameter_storage, storage_before)

        with torch.no_grad():
            model.parameter_storage.zero_()
        for name, parameter in model.named_parameters():
            expected = parameter_before[name] + (1 if name == changed_name else 0)
            torch.testing.assert_close(parameter, expected)

        model.sync_parameters_from_storage_()
        for parameter in model.parameters():
            assert torch.count_nonzero(parameter) == 0


def test_parameter_storage_round_trip_preserves_parameters() -> None:
    torch.manual_seed(0)
    packed = PackedWideResNet(depth=10, widen_factor=1, num_models=2, num_classes=5)
    normal = WideResNet(depth=10, widen_factor=1, num_classes=5)

    for model in (packed, normal):
        original = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        storage = model.parameter_storage.detach().clone()

        with torch.no_grad():
            model.parameter_storage.normal_()
        model.sync_parameters_from_storage_()
        model.parameter_storage.copy_(storage)
        model.sync_parameters_from_storage_()

        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, original[name])


def test_uses_grouped_convolutions_and_packed_batch_norm() -> None:
    model = PackedWideResNet(depth=10, widen_factor=1, num_models=3)

    convs = [module for module in model.modules() if isinstance(module, PackedConv2d)]
    norms = [module for module in model.modules() if isinstance(module, PackedBatchNorm2d)]

    assert convs
    assert all(conv.groups == 3 for conv in convs)
    assert norms
    assert all(norm.num_features == 3 * norm.local_num_features for norm in norms)
    assert all(conv.weight.is_contiguous(memory_format=torch.channels_last) for conv in convs)


def test_channels_last_weights_have_contiguous_zero_copy_storage_sync_views() -> None:
    models = [
        PackedWideResNet(depth=10, widen_factor=1, num_models=3),
        WideResNet(depth=10, widen_factor=1),
    ]

    for model in models:
        model.parameter_storage
        named_parameters = dict(model.named_parameters())
        for name, parameter_tensor in zip(
            model._parameter_names,
            model._parameter_tensors,
            strict=True,
        ):
            parameter = named_parameters[name]
            if parameter.ndim == 4:
                assert parameter.is_contiguous(memory_format=torch.channels_last)
            assert parameter_tensor.is_contiguous()
            assert _storage_ptr(parameter_tensor) == _storage_ptr(parameter)


def test_packed_model_matches_separate_single_model_forwards() -> None:
    torch.manual_seed(0)
    models = [
        PackedWideResNet(depth=10, widen_factor=1, num_models=1, num_classes=6)
        for _ in range(3)
    ]
    packed = PackedWideResNet(depth=10, widen_factor=1, num_models=3, num_classes=6)
    copy_single_models_into_packed(packed, models)
    packed.eval()
    for model in models:
        model.eval()

    x = _packed_input(4, 3)
    packed_logits = packed(x)
    separate_logits = torch.cat(
        [
            model(x[:, idx * 3 : (idx + 1) * 3].contiguous(memory_format=torch.channels_last))
            for idx, model in enumerate(models)
        ],
        dim=1,
    )

    torch.testing.assert_close(packed_logits, separate_logits, rtol=1e-5, atol=1e-5)


def test_packed_model_matches_normal_wide_resnet_forwards() -> None:
    torch.manual_seed(0)
    models = [WideResNet(depth=10, widen_factor=1, num_classes=6) for _ in range(3)]
    packed = PackedWideResNet(depth=10, widen_factor=1, num_models=3, num_classes=6)
    copy_single_models_into_packed(packed, models)
    packed.eval()
    for model in models:
        model.eval()

    x = _packed_input(4, 3)
    packed_logits = packed(x)
    separate_logits = torch.stack(
        [
            model(x[:, idx * 3 : (idx + 1) * 3].contiguous(memory_format=torch.channels_last))
            for idx, model in enumerate(models)
        ],
        dim=1,
    )

    torch.testing.assert_close(packed_logits, separate_logits, rtol=1e-5, atol=1e-5)


def test_training_packed_model_matches_separate_models_outputs_gradients_and_batch_norm() -> None:
    torch.manual_seed(0)
    models = [WideResNet(depth=10, widen_factor=1, num_classes=6) for _ in range(2)]
    packed = PackedWideResNet(depth=10, widen_factor=1, num_models=2, num_classes=6)
    copy_single_models_into_packed(packed, models)
    x = _packed_input(4, 2)

    packed_logits = packed(x)
    separate_logits = torch.stack(
        [
            model(x[:, idx * 3 : (idx + 1) * 3].contiguous(memory_format=torch.channels_last))
            for idx, model in enumerate(models)
        ],
        dim=1,
    )
    output_gradient = torch.randn_like(packed_logits)
    packed_logits.backward(output_gradient)
    separate_logits.backward(output_gradient)

    torch.testing.assert_close(packed_logits, separate_logits, rtol=1e-5, atol=1e-5)
    separate_parameters = [dict(model.named_parameters()) for model in models]
    for name, parameter in packed.named_parameters():
        assert parameter.grad is not None
        expected_gradients = []
        for model_parameters in separate_parameters:
            reference_gradient = model_parameters[name].grad
            assert reference_gradient is not None
            expected_gradients.append(_local_tensor_view(reference_gradient, 1)[0])
        torch.testing.assert_close(
            _local_tensor_view(parameter.grad, packed.num_models),
            torch.stack(expected_gradients),
            rtol=1e-5,
            atol=1e-5,
        )

    packed_modules = dict(packed.named_modules())
    separate_modules = [dict(model.named_modules()) for model in models]
    for name, module in packed_modules.items():
        if not isinstance(module, PackedBatchNorm2d):
            continue
        assert module.running_mean is not None
        assert module.running_var is not None
        assert module.num_batches_tracked is not None
        for model_idx, model_modules in enumerate(separate_modules):
            reference = model_modules[name]
            assert isinstance(reference, torch.nn.BatchNorm2d)
            start = model_idx * module.local_num_features
            end = start + module.local_num_features
            torch.testing.assert_close(module.running_mean[start:end], reference.running_mean)
            torch.testing.assert_close(module.running_var[start:end], reference.running_var)
            torch.testing.assert_close(module.num_batches_tracked, reference.num_batches_tracked)


def test_changing_one_local_model_input_does_not_change_other_outputs() -> None:
    torch.manual_seed(0)
    model = PackedWideResNet(depth=10, widen_factor=1, num_models=3, num_classes=6).eval()
    x = _packed_input(4, 3)
    changed_x = x.clone()
    changed_x[:, :3].add_(10)

    output = model(x)
    changed_output = model(changed_x)

    assert not torch.allclose(output[:, 0], changed_output[:, 0])
    torch.testing.assert_close(output[:, 1:], changed_output[:, 1:], rtol=0, atol=0)


def test_packed_parameter_storage_rows_match_normal_models_after_copy() -> None:
    torch.manual_seed(0)
    models = [WideResNet(depth=10, widen_factor=1, num_classes=6) for _ in range(3)]
    packed = PackedWideResNet(depth=10, widen_factor=1, num_models=3, num_classes=6)

    copy_single_models_into_packed(packed, models)

    for idx, model in enumerate(models):
        torch.testing.assert_close(packed.parameter_storage[idx : idx + 1], model.parameter_storage)


def test_parameter_storage_mixing_changes_outputs_after_sync() -> None:
    torch.manual_seed(0)
    model = PackedWideResNet(depth=10, widen_factor=1, num_models=2, num_classes=4)
    model.eval()
    x = _packed_input(2, 2)
    before = model(x)

    with torch.no_grad():
        model.parameter_storage.zero_()
    model.sync_parameters_from_storage_()
    after = model(x)

    assert not torch.allclose(before, after)


def test_average_returns_global_average_wide_resnet() -> None:
    torch.manual_seed(0)
    models = [WideResNet(depth=10, widen_factor=1, num_classes=6) for _ in range(3)]
    packed = PackedWideResNet(depth=10, widen_factor=1, num_models=3, num_classes=6)
    copy_single_models_into_packed(packed, models)

    averaged = packed.average()
    expected_storage = torch.stack([model.parameter_storage[0] for model in models]).mean(dim=0, keepdim=True)

    assert isinstance(averaged, WideResNet)
    torch.testing.assert_close(averaged.parameter_storage, expected_storage)
    for name, parameter in averaged.named_parameters():
        expected = torch.stack([dict(model.named_parameters())[name] for model in models]).mean(dim=0)
        torch.testing.assert_close(parameter, expected)

    existing = WideResNet(depth=10, widen_factor=1, num_classes=6)
    returned = packed.average(existing)
    assert returned is existing
    torch.testing.assert_close(existing.parameter_storage, expected_storage)


def test_average_rejects_mismatched_target() -> None:
    packed = PackedWideResNet(depth=10, widen_factor=1, num_models=2, num_classes=5)
    target = WideResNet(depth=10, widen_factor=1, num_classes=6)

    with pytest.raises(ValueError, match="target model must match"):
        packed.average(target)


def test_gradient_isolation_between_local_models() -> None:
    torch.manual_seed(0)
    model = PackedWideResNet(depth=10, widen_factor=1, num_models=2, num_classes=4)
    x = _packed_input(2, 2)

    loss = model(x)[:, 0].sum()
    loss.backward()

    classifier = model.classifier
    assert isinstance(classifier, PackedLinear)
    assert classifier.weight.grad is not None
    assert torch.count_nonzero(classifier.weight.grad[0]) > 0
    assert torch.count_nonzero(classifier.weight.grad[1]) == 0


def test_gradients_are_isolated_for_every_parameter() -> None:
    torch.manual_seed(0)
    model = PackedWideResNet(depth=10, widen_factor=1, num_models=2, num_classes=4)
    x = _packed_input(2, 2)

    model(x)[:, 0].sum().backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        local_gradients = _local_tensor_view(parameter.grad, model.num_models)
        assert torch.count_nonzero(local_gradients[1]) == 0, name


def test_batch_norm_affine_parameters_are_mixed_but_buffers_are_excluded() -> None:
    torch.manual_seed(0)
    model = PackedWideResNet(depth=10, widen_factor=1, num_models=2, num_classes=4)
    model(_packed_input(4, 2))
    buffers_before = {name: buffer.detach().clone() for name, buffer in model.named_buffers()}
    batch_norms = [module for module in model.modules() if isinstance(module, PackedBatchNorm2d)]

    with torch.no_grad():
        model.parameter_storage.zero_()
    model.sync_parameters_from_storage_()

    for batch_norm in batch_norms:
        assert torch.count_nonzero(batch_norm.weight) == 0
        assert torch.count_nonzero(batch_norm.bias) == 0
    for name, buffer in model.named_buffers():
        torch.testing.assert_close(buffer, buffers_before[name], rtol=0, atol=0)


@pytest.mark.parametrize("bias", [False, True])
def test_packed_linear_matches_independent_linear_layers(bias: bool) -> None:
    torch.manual_seed(0)
    layer = PackedLinear(num_models=3, in_features=7, out_features=5, bias=bias)
    input = torch.randn(4, 3, 7, requires_grad=True)
    reference_input = input.detach().clone().requires_grad_()
    reference_weight = layer.weight.detach().clone().requires_grad_()
    reference_bias = (
        layer.bias.detach().clone().requires_grad_() if layer.bias is not None else None
    )

    output = layer(input)
    reference_output = torch.stack(
        [
            torch.nn.functional.linear(
                reference_input[:, model_idx],
                reference_weight[model_idx],
                None if reference_bias is None else reference_bias[model_idx],
            )
            for model_idx in range(layer.num_models)
        ],
        dim=1,
    )
    output_gradient = torch.randn_like(output)
    output.backward(output_gradient)
    reference_output.backward(output_gradient)

    torch.testing.assert_close(output, reference_output)
    torch.testing.assert_close(input.grad, reference_input.grad)
    torch.testing.assert_close(layer.weight.grad, reference_weight.grad)
    if layer.bias is not None:
        assert reference_bias is not None
        torch.testing.assert_close(layer.bias.grad, reference_bias.grad)


def test_invalid_shapes_raise_clear_errors() -> None:
    model = PackedWideResNet(depth=10, widen_factor=1, num_models=2)

    with pytest.raises(ValueError, match=r"expects \[B, K\*C, H, W\]"):
        model(torch.randn(2, 2, 3, 32, 32))
    with pytest.raises(ValueError, match=r"expected K\*C=6"):
        model(_channels_last_input(2, 3))
    with pytest.raises(ValueError, match="channels-last"):
        model(torch.randn(2, 6, 32, 32))

    normal = WideResNet(depth=10, widen_factor=1)
    with pytest.raises(ValueError, match=r"expects \[B, C, H, W\]"):
        normal(torch.randn(2, 1, 3, 32, 32))
    with pytest.raises(ValueError, match="expected C=3"):
        normal(_channels_last_input(2, 1))
    with pytest.raises(ValueError, match="channels-last"):
        normal(torch.randn(2, 3, 32, 32))


def test_depth_validation() -> None:
    with pytest.raises(ValueError, match="depth = 6n \\+ 4"):
        PackedWideResNet(depth=12, widen_factor=1, num_models=1)
    with pytest.raises(ValueError, match="depth = 6n \\+ 4"):
        WideResNet(depth=12, widen_factor=1)


def test_constructor_helper() -> None:
    model = packed_wrn_28_10(num_models=2, num_classes=11)
    normal = wrn_28_10(num_classes=11)

    assert model.depth == 28
    assert model.widen_factor == 10
    assert model.num_models == 2
    assert model.num_classes == 11
    assert normal.depth == 28
    assert normal.widen_factor == 10
    assert normal.num_classes == 11
