from __future__ import annotations

import pytest
import torch
from torch import nn

from packed_resnet import MLP, PackedLinear, PackedMLP


def _assert_local_parameters_identical(model: PackedMLP) -> None:
    for parameter in model.parameters():
        local_parameters = parameter.view(model.num_models, -1)
        torch.testing.assert_close(
            local_parameters,
            local_parameters[0].unsqueeze(0).expand_as(local_parameters),
            rtol=0,
            atol=0,
        )


def _copy_into_independent_models(model: PackedMLP) -> list[MLP]:
    independent_models: list[MLP] = []
    for model_idx in range(model.num_models):
        independent_model = MLP(
            in_features=model.in_features,
            hidden_features=model.hidden_features,
            out_features=model.out_features,
            bias=model.layers[0].bias is not None,
        )
        for packed_layer, linear in zip(
            model.layers, independent_model.layers, strict=True
        ):
            with torch.no_grad():
                linear.weight.copy_(packed_layer.weight[model_idx])
                if linear.bias is not None and packed_layer.bias is not None:
                    linear.bias.copy_(packed_layer.bias[model_idx])
        independent_models.append(independent_model)
    return independent_models


def test_packed_mlp_initializes_all_local_models_identically() -> None:
    model = PackedMLP(num_models=4, in_features=7, hidden_features=(11, 5), out_features=3)

    _assert_local_parameters_identical(model)


@pytest.mark.parametrize("bias", [False, True])
def test_packed_mlp_matches_independent_models_and_gradients(bias: bool) -> None:
    torch.manual_seed(0)
    model = PackedMLP(
        num_models=3,
        in_features=7,
        hidden_features=(11, 5),
        out_features=4,
        bias=bias,
    )
    independent_models = _copy_into_independent_models(model)
    input = torch.randn(6, 3, 7, requires_grad=True)
    independent_inputs = [
        input[:, model_idx].detach().clone().requires_grad_() for model_idx in range(3)
    ]

    output = model(input)
    independent_output = torch.stack(
        [
            independent_model(independent_input)
            for independent_model, independent_input in zip(
                independent_models, independent_inputs, strict=True
            )
        ],
        dim=1,
    )
    output_gradient = torch.randn_like(output)
    output.backward(output_gradient)
    independent_output.backward(output_gradient)

    torch.testing.assert_close(output, independent_output)
    torch.testing.assert_close(
        input.grad,
        torch.stack([independent_input.grad for independent_input in independent_inputs], dim=1),
    )
    for layer_idx, packed_layer in enumerate(model.layers):
        independent_layers = [independent_model.layers[layer_idx] for independent_model in independent_models]
        assert all(isinstance(layer, nn.Linear) for layer in independent_layers)
        torch.testing.assert_close(
            packed_layer.weight.grad,
            torch.stack([layer.weight.grad for layer in independent_layers]),
        )
        if packed_layer.bias is not None:
            torch.testing.assert_close(
                packed_layer.bias.grad,
                torch.stack([layer.bias.grad for layer in independent_layers]),
            )


def test_packed_mlp_without_hidden_layers_is_one_packed_linear() -> None:
    model = PackedMLP(
        num_models=2,
        in_features=5,
        hidden_features=(),
        out_features=3,
    )
    input = torch.randn(4, 2, 5)

    assert len(model.layers) == 1
    assert len(model.activations) == 0
    assert isinstance(model.layers[0], PackedLinear)
    torch.testing.assert_close(model(input), model.layers[0](input))


def test_packed_mlp_rejects_invalid_feature_widths_and_input_shapes() -> None:
    with pytest.raises(ValueError, match="all feature widths must be >= 1"):
        PackedMLP(num_models=2, in_features=4, hidden_features=(0,), out_features=3)

    model = PackedMLP(num_models=2, in_features=4, hidden_features=(8,), out_features=3)
    with pytest.raises(ValueError, match=r"expects \[B, K, F\]"):
        model(torch.randn(2, 4))
    with pytest.raises(ValueError, match="expected K=2"):
        model(torch.randn(2, 3, 4))
    with pytest.raises(ValueError, match="expected F=4"):
        model(torch.randn(2, 2, 5))


def test_packed_mlp_supports_torch_compile() -> None:
    model = PackedMLP(num_models=2, in_features=4, hidden_features=(8,), out_features=3)
    input = torch.randn(5, 2, 4)

    expected = model(input)
    actual = torch.compile(model, backend="eager")(input)

    torch.testing.assert_close(actual, expected)


def test_mlp_forward_shape_and_without_hidden_layers() -> None:
    model = MLP(in_features=5, hidden_features=(7, 6), out_features=3)
    linear_model = MLP(in_features=5, hidden_features=(), out_features=3)
    input = torch.randn(4, 5)

    assert model(input).shape == (4, 3)
    assert len(linear_model.layers) == 1
    assert len(linear_model.activations) == 0
    torch.testing.assert_close(linear_model(input), linear_model.layers[0](input))


def test_mlp_rejects_invalid_feature_widths_and_input_shapes() -> None:
    with pytest.raises(ValueError, match="all feature widths must be >= 1"):
        MLP(in_features=4, hidden_features=(0,), out_features=3)

    model = MLP(in_features=4, hidden_features=(8,), out_features=3)
    with pytest.raises(ValueError, match=r"expects \[B, F\]"):
        model(torch.randn(2, 1, 4))
    with pytest.raises(ValueError, match="expected F=4"):
        model(torch.randn(2, 5))


def test_mlp_supports_torch_compile() -> None:
    model = MLP(in_features=4, hidden_features=(8,), out_features=3)
    input = torch.randn(5, 4)

    expected = model(input)
    actual = torch.compile(model, backend="eager")(input)

    torch.testing.assert_close(actual, expected)
