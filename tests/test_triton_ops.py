from __future__ import annotations

import copy

import pytest
import torch

from packed_resnet.layers import _batch_norm_relu
from packed_resnet.triton_ops import triton_batch_norm_relu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("training", [False, True])
def test_triton_batch_norm_relu_matches_native_forward_backward_and_state(
    training: bool,
) -> None:
    torch.manual_seed(0)
    input = torch.randn(4, 16, 16, 16, device="cuda").contiguous(
        memory_format=torch.channels_last
    )
    input.requires_grad_()
    reference_input = input.detach().clone().requires_grad_()
    batch_norm = torch.nn.BatchNorm2d(16, device="cuda").train(training)
    reference_batch_norm = copy.deepcopy(batch_norm)

    assert batch_norm.running_mean is not None
    assert batch_norm.running_var is not None
    assert batch_norm.num_batches_tracked is not None
    assert batch_norm.weight is not None
    assert batch_norm.bias is not None
    output = triton_batch_norm_relu(
        input,
        batch_norm.weight,
        batch_norm.bias,
        batch_norm.running_mean,
        batch_norm.running_var,
        batch_norm.num_batches_tracked,
        training=training,
        momentum=0.1,
        eps=batch_norm.eps,
    )
    reference_output = torch.relu(reference_batch_norm(reference_input))
    output_gradient = torch.randn_like(output)
    output.backward(output_gradient)
    reference_output.backward(output_gradient)

    assert output.is_contiguous(memory_format=torch.channels_last)
    torch.testing.assert_close(output, reference_output, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(input.grad, reference_input.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(batch_norm.weight.grad, reference_batch_norm.weight.grad)
    torch.testing.assert_close(batch_norm.bias.grad, reference_batch_norm.bias.grad)
    torch.testing.assert_close(batch_norm.running_mean, reference_batch_norm.running_mean)
    torch.testing.assert_close(batch_norm.running_var, reference_batch_norm.running_var)
    torch.testing.assert_close(
        batch_norm.num_batches_tracked,
        reference_batch_norm.num_batches_tracked,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_triton_batch_norm_relu_supports_bfloat16() -> None:
    input = torch.randn(2, 8, 8, 8, device="cuda", dtype=torch.bfloat16).contiguous(
        memory_format=torch.channels_last
    )
    input.requires_grad_()
    batch_norm = torch.nn.BatchNorm2d(8, device="cuda")

    output = _batch_norm_relu(input, batch_norm)
    output.sum().backward()

    assert output.dtype == torch.bfloat16
    assert input.grad is not None
    assert batch_norm.weight.grad is not None
