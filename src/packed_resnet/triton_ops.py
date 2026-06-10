from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _batch_norm_relu_training_forward_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    running_mean_ptr,
    running_var_ptr,
    num_batches_tracked_ptr,
    output_ptr,
    save_mean_ptr,
    save_invstd_ptr,
    rows: tl.constexpr,
    channels: tl.constexpr,
    momentum: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    channel = tl.program_id(0)
    row_offsets = tl.arange(0, BLOCK_SIZE)
    mask = row_offsets < rows
    offsets = row_offsets * channels + channel
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(values, axis=0) / rows
    centered = tl.where(mask, values - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / rows
    invstd = tl.rsqrt(variance + eps)
    tl.store(save_mean_ptr + channel, mean)
    tl.store(save_invstd_ptr + channel, invstd)

    old_mean = tl.load(running_mean_ptr + channel).to(tl.float32)
    old_var = tl.load(running_var_ptr + channel).to(tl.float32)
    unbiased_var = variance * rows / (rows - 1) if rows > 1 else variance
    tl.store(running_mean_ptr + channel, (1.0 - momentum) * old_mean + momentum * mean)
    tl.store(running_var_ptr + channel, (1.0 - momentum) * old_var + momentum * unbiased_var)
    if channel == 0:
        tl.atomic_add(num_batches_tracked_ptr, 1)

    weight = tl.load(weight_ptr + channel).to(tl.float32)
    bias = tl.load(bias_ptr + channel).to(tl.float32)
    output = tl.maximum((values - mean) * invstd * weight + bias, 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def _batch_norm_relu_eval_forward_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    running_mean_ptr,
    running_var_ptr,
    output_ptr,
    save_mean_ptr,
    save_invstd_ptr,
    numel: tl.constexpr,
    channels: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    channel = offsets % channels
    mean = tl.load(running_mean_ptr + channel, mask=mask).to(tl.float32)
    invstd = tl.rsqrt(tl.load(running_var_ptr + channel, mask=mask).to(tl.float32) + eps)
    weight = tl.load(weight_ptr + channel, mask=mask).to(tl.float32)
    bias = tl.load(bias_ptr + channel, mask=mask).to(tl.float32)
    values = tl.load(input_ptr + offsets, mask=mask).to(tl.float32)
    output = tl.maximum((values - mean) * invstd * weight + bias, 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)

    stats_mask = offsets < channels
    tl.store(save_mean_ptr + offsets, mean, mask=stats_mask)
    tl.store(save_invstd_ptr + offsets, invstd, mask=stats_mask)


class _BatchNormReLU(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        weight: Tensor,
        bias: Tensor,
        running_mean: Tensor,
        running_var: Tensor,
        num_batches_tracked: Tensor,
        training: bool,
        momentum: float,
        eps: float,
    ) -> Tensor:
        channels = input.shape[1]
        rows = input.numel() // channels
        output = torch.empty_like(input, memory_format=torch.preserve_format)
        save_mean = torch.empty(channels, device=input.device, dtype=torch.float32)
        save_invstd = torch.empty(channels, device=input.device, dtype=torch.float32)
        if training:
            block_size = triton.next_power_of_2(rows)
            _batch_norm_relu_training_forward_kernel[(channels,)](
                input,
                weight,
                bias,
                running_mean,
                running_var,
                num_batches_tracked,
                output,
                save_mean,
                save_invstd,
                rows=rows,
                channels=channels,
                momentum=momentum,
                eps=eps,
                BLOCK_SIZE=block_size,
            )
        else:
            block_size = 256
            grid = (triton.cdiv(input.numel(), block_size),)
            _batch_norm_relu_eval_forward_kernel[grid](
                input,
                weight,
                bias,
                running_mean,
                running_var,
                output,
                save_mean,
                save_invstd,
                numel=input.numel(),
                channels=channels,
                eps=eps,
                BLOCK_SIZE=block_size,
            )
        ctx.training = training
        ctx.eps = eps
        ctx.save_for_backward(
            input,
            weight,
            running_mean,
            running_var,
            save_mean,
            save_invstd,
            output,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor | None, ...]:
        input, weight, running_mean, running_var, save_mean, save_invstd, output = ctx.saved_tensors
        grad_relu = torch.where(output > 0, grad_output, 0)
        grad_input, grad_weight, grad_bias = torch.ops.aten.native_batch_norm_backward(
            grad_relu,
            input,
            weight,
            running_mean,
            running_var,
            save_mean,
            save_invstd,
            ctx.training,
            ctx.eps,
            (True, True, True),
        )
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None


def triton_batch_norm_relu(
    input: Tensor,
    weight: Tensor,
    bias: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    num_batches_tracked: Tensor,
    *,
    training: bool,
    momentum: float,
    eps: float,
) -> Tensor:
    """Apply fused channels-last BatchNorm2d and ReLU using Triton."""

    return _BatchNormReLU.apply(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        num_batches_tracked,
        training,
        momentum,
        eps,
    )
