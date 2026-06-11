# packed-resnet

Packed CIFAR-style Wide ResNet modules for simulating multiple independent local
models in one PyTorch module.

```python
import torch
from packed_resnet import packed_wrn_28_10, wrn_28_10

single_model = wrn_28_10(num_classes=10)
model = packed_wrn_28_10(num_models=4, num_classes=10)
x = torch.randn(8, 4 * 3, 32, 32).contiguous(memory_format=torch.channels_last)
logits = model(x)
assert logits.shape == (8, 4, 10)
```

Wide ResNet inputs use channels-last contiguous `[B, C, H, W]` tensors. Packed
Wide ResNet inputs use channels-last contiguous `[B, K * C, H, W]` tensors,
where each contiguous group of `C` channels belongs to one local model and `K`
is fixed when the model is created. Convolutions use channels-last weights and
`groups=K`, BatchNorm uses `BatchNorm2d(K * C)`, and the final classifier is
packed with one independent weight matrix per local model.
Packed models initialize model 0 once and broadcast its parameters to every
other local model, so all local models start from identical parameters.

## Numerical Precision

Packed and standalone models are mathematically equivalent, but they are not
expected to be bitwise identical on CUDA. Packed convolutions use one grouped
convolution while standalone models use separate convolutions, and packed
BatchNorm processes `K * C` channels in one operation. cuDNN may select
different kernels and reduction orders for these tensor shapes.

In FP32, the resulting output and gradient differences are normally close to
floating-point roundoff. TF32 and BF16 increase the differences because tensor
core multiplications use reduced-precision inputs. Tests should therefore use
precision-appropriate tolerances while continuing to require exact isolation:
changing or differentiating through one local model must not affect another
local model. Reduced precision can also produce a small downward bias in
BatchNorm running variances relative to an IEEE FP32 reference.

For CUDA Wide ResNet benchmarks, convolution precision is controlled by
`torch.backends.cudnn.allow_tf32`, while
`torch.set_float32_matmul_precision()` controls matrix multiplications such as
the classifier. Changing only the matmul setting may have little effect because
the benchmark is dominated by convolutions. The benchmark script configures
both controls: non-AMP runs use TF32, and `--amp bf16` uses BF16 autocast with
TF32 disabled for FP32 fallback operations.

## Packed Distributed Dataloaders

`create_dataloader` downloads MNIST, CIFAR10, or CIFAR100 and keeps the complete
split as `float32` tensors on CUDA when available. Each epoch exactly reproduces
PyTorch `DistributedSampler` sharding for the selected simulated ranks. The
factory always downloads missing data under `./data` by default and applies
standard dataset normalization:

```python
from packed_resnet import create_dataloader

loader = create_dataloader(
    "cifar10",
    local_batch_size=64,       # Per-rank batch size.
    world_size=16,
    ranks=[2, 5, 9, 12],
    base_seed=0,
)

for epoch in range(10):
    loader.set_epoch(epoch)
    for images, targets in loader:
        # images: channels-last [64, 4 * 3, 32, 32]
        # targets: [64, 4]
        ...
```

Training loaders shuffle and apply deterministic per-rank CIFAR random crops
and horizontal flips by default. MNIST augmentation is disabled by default;
passing `augment=True` applies deterministic random crops without horizontal
flips. Test loaders disable augmentation. Standard dataset normalization is
enabled by default.

Pass `packed=False` for compatibility with a normal single-worker model. This
mode requires exactly one rank and returns images shaped `[B, C, H, W]` with
targets shaped `[B]`. Images are channels-last contiguous by default; pass
`channels_last=False` for standard contiguous NCHW output.

Benchmark complete dataloader epochs for a world size and number of packed
ranks with:

```bash
uv run python tests/benchmark_dataloader.py --world-size 32 --num-ranks 8
uv run python tests/benchmark_dataloader.py --dataset cifar100 --world-size 64 --num-ranks 16 --local-batch-size 128
uv run python tests/benchmark_dataloader.py --world-size 32 --num-ranks 8 --device cpu --no-augment
```

Dataset download and initial GPU materialization happen before timing. The
reported epoch time includes deterministic sampling, indexing, augmentation,
normalization, packing, and synchronization of all resulting GPU work.

## MLP Models

`MLP` provides a standard single-model MLP, while `PackedMLP` applies
independent MLPs to inputs shaped `[B, K, F]`:

```python
from packed_resnet import MLP, PackedMLP

single_mlp = MLP(
    in_features=128,
    hidden_features=(256, 256),
    out_features=10,
)

mlp = PackedMLP(
    num_models=4,
    in_features=128,
    hidden_features=(256, 256),
    out_features=10,
)
logits = mlp(torch.randn(8, 4, 128))
assert logits.shape == (8, 4, 10)
```

Activations are applied after hidden layers only. Pass `hidden_features=()` for
a single packed linear projection.

## Parameter Storage and Mixing

Both `PackedWideResNet` and `WideResNet` provide a contiguous
`parameter_storage` tensor for decentralized parameter mixing:

- packed models use `[K, D]`
- normal models use `[1, D]`
- each parameter segment is padded to a 64-element boundary
- BatchNorm running statistics are not included
- storage is materialized lazily and is excluded from `state_dict()` and
  `torch.export`

All trainable parameters own memory independently from `parameter_storage`.
Synchronize explicitly around mixing:

```python
packed.sync_storage_from_parameters_()

with torch.no_grad():
    packed.parameter_storage.copy_(mixing_matrix @ packed.parameter_storage)

packed.sync_parameters_from_storage_()
```

Both sync directions copy every trainable parameter using cached physical-layout
tensor views and `torch._foreach_copy_`. Channels-last convolution weights are
copied directly without first materializing contiguous-format copies. The same
sync APIs are available on `WideResNet`.

To create a standard WideResNet whose parameters are the global average of the
packed local models:

```python
averaged = packed.average()

target = wrn_28_10(num_classes=10)
packed.average(target)  # updates and returns target
```

## Development

```bash
uv sync
uv run pytest
uv run ruff check
```

## GPU Timing Benchmarks

```bash
uv run python tests/benchmark_gpu_timing.py
uv run python tests/benchmark_gpu_timing.py --model wrn28-10
uv run python tests/benchmark_gpu_timing.py --model all
uv run python tests/benchmark_gpu_timing.py --amp bf16
uv run python tests/benchmark_gpu_timing.py --amp bf16 --compile
uv run python tests/benchmark_gpu_timing.py --include-optimizer-step
uv run python tests/benchmark_gpu_timing.py --include-storage-sync
uv run python tests/benchmark_gpu_timing.py --compile --batch-sizes 16 --num-models 8
```

The benchmark times WRN16-8 with TF32 by default, with WRN28-10 available via
`--model wrn28-10`. Use `--model all` to run both. Each selected model is timed
for forward+backward on CUDA across:

- normal single-model batches: `16`, `32`, `64`
- packed local batch sizes: `16`, `32`, `64`
- packed model counts: `8`, `16`, `32`

Use `--amp bf16` to replace TF32 with BF16 CUDA autocast. Use `--compile` to run
`torch.compile` before warmup; `--compile-mode` accepts `default`,
`reduce-overhead`, or `max-autotune`. Use at least one warmup step with
`--compile` to keep first-time compilation outside the timed region. For quick
smoke runs, `--batch-sizes` and `--num-models` accept comma-separated subsets of
the default grid. Use `--include-storage-sync` to include one
`sync_storage_from_parameters_()` call and one `sync_parameters_from_storage_()`
call in every timed step.
