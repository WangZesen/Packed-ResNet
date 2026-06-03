# packed-resnet

Packed CIFAR-style Wide ResNet modules for simulating multiple independent local
models in one PyTorch module.

```python
import torch
from packed_resnet import packed_wrn_28_10, wrn_28_10

single_model = wrn_28_10(num_classes=10)
model = packed_wrn_28_10(num_models=4, num_classes=10)
x = torch.randn(8, 4, 3, 32, 32)
logits = model(x)
assert logits.shape == (8, 4, 10)
```

Inputs use `[B, K, C, H, W]`, where `K` is fixed when the model is created.
Convolutional activations are viewed internally as `[B, K * C, H, W]`.
Convolutions use `groups=K`, BatchNorm uses `BatchNorm2d(K * C)`, and the final
classifier is packed with one independent weight matrix per local model.

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
uv run python tests/benchmark_gpu_timing.py --compile --batch-sizes 16 --num-models 8
```

The benchmark times WRN16-8 by default, with WRN28-10 available via
`--model wrn28-10`. Use `--model all` to run both. Each selected model is timed
for forward+backward on CUDA across:

- normal single-model batches: `16`, `32`, `64`
- packed local batch sizes: `16`, `32`, `64`
- packed model counts: `8`, `16`, `32`

Use `--amp bf16` to time BF16 CUDA autocast. Use `--compile` to run
`torch.compile` before warmup; `--compile-mode` accepts `default`,
`reduce-overhead`, or `max-autotune`. Use at least one warmup step with
`--compile` to keep first-time compilation outside the timed region. For quick
smoke runs, `--batch-sizes` and `--num-models` accept comma-separated subsets of
the default grid.
