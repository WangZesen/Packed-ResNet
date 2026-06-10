from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from packed_resnet import PackedDataLoader, create_dataloader


@dataclass(frozen=True)
class BenchmarkResult:
    epoch_times_s: tuple[float, ...]
    batches_per_epoch: int
    local_samples_per_epoch: int

    @property
    def mean_s(self) -> float:
        return statistics.fmean(self.epoch_times_s)

    @property
    def samples_per_second(self) -> float:
        return self.local_samples_per_epoch / self.mean_s

    @property
    def batches_per_second(self) -> float:
        return self.batches_per_epoch / self.mean_s


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _iterate_epoch(loader: PackedDataLoader) -> tuple[int, int]:
    batches = 0
    local_samples = 0
    for images, targets in loader:
        batches += 1
        local_samples += targets.numel()
        # Keep references live until both tensors have been produced.
        del images, targets
    return batches, local_samples


def benchmark_loader(
    loader: PackedDataLoader,
    *,
    warmup_epochs: int,
    timing_epochs: int,
) -> BenchmarkResult:
    for epoch in range(warmup_epochs):
        loader.set_epoch(epoch)
        _iterate_epoch(loader)
    _synchronize(loader.device)

    elapsed: list[float] = []
    expected_counts: tuple[int, int] | None = None
    for epoch in range(warmup_epochs, warmup_epochs + timing_epochs):
        loader.set_epoch(epoch)
        _synchronize(loader.device)
        start = time.perf_counter()
        counts = _iterate_epoch(loader)
        _synchronize(loader.device)
        elapsed.append(time.perf_counter() - start)
        if expected_counts is None:
            expected_counts = counts
        elif counts != expected_counts:
            raise RuntimeError(f"epoch batch/sample counts changed from {expected_counts} to {counts}")

    assert expected_counts is not None
    return BenchmarkResult(
        epoch_times_s=tuple(elapsed),
        batches_per_epoch=expected_counts[0],
        local_samples_per_epoch=expected_counts[1],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark complete PackedDataLoader iteration epochs.",
    )
    parser.add_argument("--dataset", choices=("mnist", "cifar10", "cifar100"), default="cifar10")
    parser.add_argument("--root", type=Path, default=Path("./data"))
    parser.add_argument("--local-batch-size", type=int, default=64, help="Per-rank batch size.")
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument(
        "--num-ranks",
        type=int,
        required=True,
        help="Number of simulated ranks to pack, selected as ranks 0 through num-ranks - 1.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--timing-epochs", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the loader's split-dependent augmentation default.",
    )
    parser.add_argument("--sampler-drop-last", action="store_true")
    parser.add_argument("--drop-last", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.world_size < 1:
        raise ValueError("--world-size must be >= 1")
    if args.num_ranks < 1 or args.num_ranks > args.world_size:
        raise ValueError("--num-ranks must be in [1, world-size]")
    if args.local_batch_size < 1:
        raise ValueError("--local-batch-size must be >= 1")
    if args.warmup_epochs < 0:
        raise ValueError("--warmup-epochs must be >= 0")
    if args.timing_epochs < 1:
        raise ValueError("--timing-epochs must be >= 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    device = None if args.device == "auto" else torch.device(args.device)
    ranks = tuple(range(args.num_ranks))
    loader = create_dataloader(
        args.dataset,
        root=args.root,
        local_batch_size=args.local_batch_size,
        world_size=args.world_size,
        ranks=ranks,
        base_seed=args.seed,
        train=args.train,
        channels_last=args.channels_last,
        augment=args.augment,
        device=device,
        sampler_drop_last=args.sampler_drop_last,
        drop_last=args.drop_last,
    )

    print(f"dataset: {args.dataset} ({'train' if args.train else 'test'})")
    print(f"device: {loader.device}")
    if loader.device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(loader.device)}")
    print(f"world_size: {args.world_size}, ranks: {args.num_ranks}, local_batch_size: {args.local_batch_size}")
    print(
        f"shuffle: {loader.shuffle}, augment: {loader.augment}, normalize: {loader.normalize}, "
        f"channels_last: {loader.channels_last}"
    )
    print(f"batches_per_epoch: {len(loader)}, samples_per_rank: {loader.num_samples}")
    print(f"warmup_epochs: {args.warmup_epochs}, timing_epochs: {args.timing_epochs}")

    result = benchmark_loader(loader, warmup_epochs=args.warmup_epochs, timing_epochs=args.timing_epochs)
    print()
    print(f"mean_epoch_s: {result.mean_s:.6f}")
    print(f"median_epoch_s: {statistics.median(result.epoch_times_s):.6f}")
    print(f"min_epoch_s: {min(result.epoch_times_s):.6f}")
    print(f"max_epoch_s: {max(result.epoch_times_s):.6f}")
    print(f"batches_per_second: {result.batches_per_second:.2f}")
    print(f"local_samples_per_second: {result.samples_per_second:.2f}")
    print(f"local_samples_per_epoch: {result.local_samples_per_epoch}")


if __name__ == "__main__":
    main()
