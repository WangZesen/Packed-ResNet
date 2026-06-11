from __future__ import annotations

import argparse
import gc
import statistics
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from packed_resnet import PackedWideResNet, WideResNet


BATCH_SIZES = (16, 32, 64)
NUM_MODELS = (8, 16, 32)
IMAGE_SHAPE = (3, 32, 32)
NUM_CLASSES = 10
MODEL_CONFIGS = {
    "wrn16-8": (16, 8),
    "wrn28-10": (28, 10),
}


@dataclass(frozen=True)
class ModelConfig:
    name: str
    depth: int
    widen_factor: int


@dataclass(frozen=True)
class BenchmarkResult:
    model_name: str
    name: str
    precision: str
    compile_mode: str
    optimizer_step: str
    storage_sync: str
    local_batch_size: int
    num_models: int
    global_batch_size: int
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    samples_per_second: float
    peak_memory_mb: float


def _single_loss(logits: Tensor, target: Tensor) -> Tensor:
    return F.cross_entropy(logits, target)


def _packed_loss(logits: Tensor, target: Tensor) -> Tensor:
    batch_size, num_models, num_classes = logits.shape
    return F.cross_entropy(logits.reshape(batch_size * num_models, num_classes), target.reshape(-1))


def _configure_precision(amp: str) -> torch.dtype | None:
    use_tf32 = amp != "bf16"
    torch.set_float32_matmul_precision("high" if use_tf32 else "highest")
    torch.backends.cudnn.allow_tf32 = use_tf32
    return torch.bfloat16 if amp == "bf16" else None


def _time_forward_backward(
    *,
    model_name: str,
    name: str,
    amp_dtype: torch.dtype | None,
    compile_mode: str | None,
    model: WideResNet | PackedWideResNet,
    input: Tensor,
    target: Tensor,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    local_batch_size: int,
    num_models: int,
    warmup_steps: int,
    timing_steps: int,
    include_optimizer_step: bool,
    include_storage_sync: bool,
) -> BenchmarkResult:
    model.train()
    forward_model: nn.Module = model
    if compile_mode is not None:
        forward_model = torch.compile(model, mode=compile_mode)  # type: ignore[assignment]
    optimizer = torch.optim.SGD(forward_model.parameters(), lr=0.01)

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            loss = loss_fn(forward_model(input), target)
        loss.backward()
        if include_optimizer_step:
            optimizer.step()
        if include_storage_sync:
            model.sync_storage_from_parameters_()
            model.sync_parameters_from_storage_()

    for _ in range(warmup_steps):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    elapsed_ms: list[float] = []
    for _ in range(timing_steps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        step()
        end.record()
        torch.cuda.synchronize()
        elapsed_ms.append(start.elapsed_time(end))

    global_batch_size = local_batch_size * num_models
    mean_ms = statistics.fmean(elapsed_ms)
    return BenchmarkResult(
        model_name=model_name,
        name=name,
        precision="bf16" if amp_dtype is torch.bfloat16 else "tf32",
        compile_mode=compile_mode or "eager",
        optimizer_step="yes" if include_optimizer_step else "no",
        storage_sync="yes" if include_storage_sync else "no",
        local_batch_size=local_batch_size,
        num_models=num_models,
        global_batch_size=global_batch_size,
        mean_ms=mean_ms,
        median_ms=statistics.median(elapsed_ms),
        min_ms=min(elapsed_ms),
        max_ms=max(elapsed_ms),
        samples_per_second=global_batch_size * 1000.0 / mean_ms,
        peak_memory_mb=torch.cuda.max_memory_allocated() / 1024**2,
    )


def _cleanup_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def benchmark_single_models(
    *,
    model_config: ModelConfig,
    device: torch.device,
    batch_sizes: tuple[int, ...],
    amp_dtype: torch.dtype | None,
    compile_mode: str | None,
    warmup_steps: int,
    timing_steps: int,
    include_optimizer_step: bool,
    include_storage_sync: bool,
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    channels, height, width = IMAGE_SHAPE
    for batch_size in batch_sizes:
        _cleanup_cuda()
        model = WideResNet(
            depth=model_config.depth,
            widen_factor=model_config.widen_factor,
            num_classes=NUM_CLASSES,
            in_channels=channels,
        ).to(device)
        input = torch.randn(batch_size, channels, height, width, device=device).contiguous(
            memory_format=torch.channels_last
        )
        target = torch.randint(NUM_CLASSES, (batch_size,), device=device)
        results.append(
            _time_forward_backward(
                model_name=model_config.name,
                name="single",
                amp_dtype=amp_dtype,
                compile_mode=compile_mode,
                model=model,
                input=input,
                target=target,
                loss_fn=_single_loss,
                local_batch_size=batch_size,
                num_models=1,
                warmup_steps=warmup_steps,
                timing_steps=timing_steps,
                include_optimizer_step=include_optimizer_step,
                include_storage_sync=include_storage_sync,
            )
        )
        del model, input, target
    return results


def benchmark_packed_models(
    *,
    model_config: ModelConfig,
    device: torch.device,
    batch_sizes: tuple[int, ...],
    num_models_list: tuple[int, ...],
    amp_dtype: torch.dtype | None,
    compile_mode: str | None,
    warmup_steps: int,
    timing_steps: int,
    include_optimizer_step: bool,
    include_storage_sync: bool,
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    channels, height, width = IMAGE_SHAPE
    for batch_size in batch_sizes:
        for num_models in num_models_list:
            _cleanup_cuda()
            model = PackedWideResNet(
                depth=model_config.depth,
                widen_factor=model_config.widen_factor,
                num_models=num_models,
                num_classes=NUM_CLASSES,
                in_channels=channels,
            ).to(device)
            input = torch.randn(
                batch_size,
                num_models * channels,
                height,
                width,
                device=device,
            ).contiguous(memory_format=torch.channels_last)
            target = torch.randint(NUM_CLASSES, (batch_size, num_models), device=device)
            results.append(
                _time_forward_backward(
                    model_name=model_config.name,
                    name="packed",
                    amp_dtype=amp_dtype,
                    compile_mode=compile_mode,
                    model=model,
                    input=input,
                    target=target,
                    loss_fn=_packed_loss,
                    local_batch_size=batch_size,
                    num_models=num_models,
                    warmup_steps=warmup_steps,
                    timing_steps=timing_steps,
                    include_optimizer_step=include_optimizer_step,
                    include_storage_sync=include_storage_sync,
                )
            )
            del model, input, target
    return results


def _print_results(results: list[BenchmarkResult]) -> None:
    header = (
        "model",
        "case",
        "prec",
        "compile",
        "optim",
        "sync",
        "local_B",
        "K",
        "global_B",
        "mean_ms",
        "median_ms",
        "min_ms",
        "max_ms",
        "samples/s",
        "peak_MB",
    )
    print(
        f"{header[0]:<8} {header[1]:<8} {header[2]:<5} {header[3]:<16} "
        f"{header[4]:<5} {header[5]:<5} {header[6]:>8} {header[7]:>4} {header[8]:>9} "
        f"{header[9]:>10} {header[10]:>10} {header[11]:>10} {header[12]:>10} "
        f"{header[13]:>12} {header[14]:>10}"
    )
    for result in results:
        print(
            f"{result.model_name:<8} {result.name:<8} {result.precision:<5} "
            f"{result.compile_mode:<16} {result.optimizer_step:<5} {result.storage_sync:<5} "
            f"{result.local_batch_size:>8} {result.num_models:>4} {result.global_batch_size:>9} "
            f"{result.mean_ms:>10.3f} {result.median_ms:>10.3f} "
            f"{result.min_ms:>10.3f} {result.max_ms:>10.3f} "
            f"{result.samples_per_second:>12.1f} {result.peak_memory_mb:>10.1f}"
        )


def _parse_int_choices(value: str, *, valid_values: tuple[int, ...], name: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{name} must be a comma-separated integer list") from error
    if not parsed:
        raise argparse.ArgumentTypeError(f"{name} must not be empty")
    invalid = sorted(set(parsed).difference(valid_values))
    if invalid:
        valid = ", ".join(str(item) for item in valid_values)
        raise argparse.ArgumentTypeError(f"{name} contains unsupported values {invalid}; valid: {valid}")
    return parsed


def _batch_sizes_arg(value: str) -> tuple[int, ...]:
    return _parse_int_choices(value, valid_values=BATCH_SIZES, name="--batch-sizes")


def _num_models_arg(value: str) -> tuple[int, ...]:
    return _parse_int_choices(value, valid_values=NUM_MODELS, name="--num-models")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU timing benchmarks for Wide ResNet models.")
    parser.add_argument(
        "--model",
        choices=(*MODEL_CONFIGS.keys(), "all"),
        default="wrn16-8",
        help="Wide ResNet variant to benchmark.",
    )
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--timing-steps", type=int, default=30)
    parser.add_argument(
        "--batch-sizes",
        type=_batch_sizes_arg,
        default=BATCH_SIZES,
        help="Comma-separated local batch sizes to run. Defaults to 16,32,64.",
    )
    parser.add_argument(
        "--num-models",
        type=_num_models_arg,
        default=NUM_MODELS,
        help="Comma-separated packed model counts to run. Defaults to 8,16,32.",
    )
    parser.add_argument(
        "--amp",
        choices=("none", "bf16"),
        default="none",
        help="Enable CUDA AMP autocast for forward/loss computation. Non-AMP runs use TF32.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile each model with torch.compile before benchmark warmup.",
    )
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
        help="torch.compile mode used when --compile is enabled.",
    )
    parser.add_argument(
        "--include-optimizer-step",
        action="store_true",
        help="Include optimizer.step() after backward in each benchmarked step.",
    )
    parser.add_argument(
        "--include-storage-sync",
        action="store_true",
        help=(
            "Include one sync_storage_from_parameters_() call and one "
            "sync_parameters_from_storage_() call in each benchmarked step."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be >= 0")
    if args.timing_steps < 1:
        raise ValueError("--timing-steps must be >= 1")
    if args.amp == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 AMP was requested, but this CUDA device does not support BF16.")

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    amp_dtype = _configure_precision(args.amp)
    precision = "bf16 AMP" if amp_dtype is torch.bfloat16 else "TF32"
    compile_mode = args.compile_mode if args.compile else None
    model_names = MODEL_CONFIGS.keys() if args.model == "all" else (args.model,)
    model_configs = [
        ModelConfig(name=model_name, depth=depth, widen_factor=widen_factor)
        for model_name in model_names
        for depth, widen_factor in (MODEL_CONFIGS[model_name],)
    ]
    print(f"device: {torch.cuda.get_device_name(device)}")
    print(f"torch: {torch.__version__}")
    print(f"models: {', '.join(config.name for config in model_configs)}, input: {IMAGE_SHAPE}")
    print(f"precision: {precision}, compile: {compile_mode or 'eager'}")
    print(f"include_optimizer_step: {args.include_optimizer_step}")
    print(f"include_storage_sync: {args.include_storage_sync}")
    print(f"batch_sizes: {args.batch_sizes}, num_models: {args.num_models}")
    print(f"warmup_steps: {args.warmup_steps}, timing_steps: {args.timing_steps}")

    results: list[BenchmarkResult] = []
    for model_config in model_configs:
        results.extend(
            benchmark_single_models(
                model_config=model_config,
                device=device,
                batch_sizes=args.batch_sizes,
                amp_dtype=amp_dtype,
                compile_mode=compile_mode,
                warmup_steps=args.warmup_steps,
                timing_steps=args.timing_steps,
                include_optimizer_step=args.include_optimizer_step,
                include_storage_sync=args.include_storage_sync,
            )
        )
        results.extend(
            benchmark_packed_models(
                model_config=model_config,
                device=device,
                batch_sizes=args.batch_sizes,
                num_models_list=args.num_models,
                amp_dtype=amp_dtype,
                compile_mode=compile_mode,
                warmup_steps=args.warmup_steps,
                timing_steps=args.timing_steps,
                include_optimizer_step=args.include_optimizer_step,
                include_storage_sync=args.include_storage_sync,
            )
        )
    _print_results(results)


if __name__ == "__main__":
    main()
