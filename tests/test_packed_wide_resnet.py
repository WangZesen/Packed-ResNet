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


def _num_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _storage_ptr(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


def test_forward_shape() -> None:
    model = PackedWideResNet(depth=10, widen_factor=1, num_models=3, num_classes=7)
    x = torch.randn(2, 3, 3, 32, 32)

    logits = model(x)

    assert logits.shape == (2, 3, 7)


def test_wide_resnet_forward_shape() -> None:
    model = WideResNet(depth=10, widen_factor=1, num_classes=7)
    x = torch.randn(2, 3, 32, 32)

    logits = model(x)

    assert logits.shape == (2, 7)


def test_parameter_count_scales_exactly_with_num_models() -> None:
    single = PackedWideResNet(depth=10, widen_factor=2, num_models=1, num_classes=5)
    normal = WideResNet(depth=10, widen_factor=2, num_classes=5)
    packed = PackedWideResNet(depth=10, widen_factor=2, num_models=4, num_classes=5)

    assert _num_parameters(packed) == 4 * _num_parameters(single)
    assert _num_parameters(packed) == 4 * _num_parameters(normal)


def test_parameter_storage_shapes_state_and_alignment() -> None:
    packed = PackedWideResNet(depth=10, widen_factor=1, num_models=3, num_classes=5)
    normal = WideResNet(depth=10, widen_factor=1, num_classes=5)

    assert packed.parameter_storage.shape == (3, packed.parameter_storage_numel())
    assert normal.parameter_storage.shape == (1, normal.parameter_storage_numel())
    assert packed.parameter_storage.shape[1] == normal.parameter_storage.shape[1]
    assert "parameter_storage" in packed.state_dict()
    assert "parameter_storage" in normal.state_dict()

    for model in (packed, normal):
        for entry in model.parameter_storage_layout():
            assert entry.start % 64 == 0
            assert entry.padded_numel % 64 == 0
            padding = model.parameter_storage[:, entry.start + entry.numel : entry.start + entry.padded_numel]
            assert torch.count_nonzero(padding) == 0


def test_parameter_storage_views_for_supported_parameters() -> None:
    packed = PackedWideResNet(depth=10, widen_factor=1, num_models=2, num_classes=5)
    normal = WideResNet(depth=10, widen_factor=1, num_classes=5)

    assert _storage_ptr(packed.classifier.weight) == _storage_ptr(packed.parameter_storage)
    assert packed.classifier.bias is not None
    assert _storage_ptr(packed.classifier.bias) == _storage_ptr(packed.parameter_storage)
    assert _storage_ptr(normal.bn.weight) == _storage_ptr(normal.parameter_storage)
    assert _storage_ptr(normal.bn.bias) == _storage_ptr(normal.parameter_storage)
    assert _storage_ptr(normal.classifier.weight) == _storage_ptr(normal.parameter_storage)
    assert normal.classifier.bias is not None
    assert _storage_ptr(normal.classifier.bias) == _storage_ptr(normal.parameter_storage)

    converted = packed.to(dtype=torch.float64)
    assert converted.parameter_storage.dtype == torch.float64
    assert _storage_ptr(converted.classifier.weight) == _storage_ptr(converted.parameter_storage)


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

    x = torch.randn(4, 3, 3, 32, 32)
    packed_logits = packed(x)
    separate_logits = torch.cat(
        [model(x[:, idx : idx + 1]) for idx, model in enumerate(models)],
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

    x = torch.randn(4, 3, 3, 32, 32)
    packed_logits = packed(x)
    separate_logits = torch.stack(
        [model(x[:, idx]) for idx, model in enumerate(models)],
        dim=1,
    )

    torch.testing.assert_close(packed_logits, separate_logits, rtol=1e-5, atol=1e-5)


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
    x = torch.randn(2, 2, 3, 32, 32)
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
    x = torch.randn(2, 2, 3, 32, 32)

    loss = model(x)[:, 0].sum()
    loss.backward()

    classifier = model.classifier
    assert isinstance(classifier, PackedLinear)
    assert classifier.weight.grad is not None
    assert torch.count_nonzero(classifier.weight.grad[0]) > 0
    assert torch.count_nonzero(classifier.weight.grad[1]) == 0


def test_invalid_shapes_raise_clear_errors() -> None:
    model = PackedWideResNet(depth=10, widen_factor=1, num_models=2)

    with pytest.raises(ValueError, match=r"expects \[B, K, C, H, W\]"):
        model(torch.randn(2, 3, 32, 32))
    with pytest.raises(ValueError, match="expected K=2"):
        model(torch.randn(2, 3, 3, 32, 32))
    with pytest.raises(ValueError, match="expected C=3"):
        model(torch.randn(2, 2, 1, 32, 32))

    normal = WideResNet(depth=10, widen_factor=1)
    with pytest.raises(ValueError, match=r"expects \[B, C, H, W\]"):
        normal(torch.randn(2, 1, 3, 32, 32))
    with pytest.raises(ValueError, match="expected C=3"):
        normal(torch.randn(2, 1, 32, 32))


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
