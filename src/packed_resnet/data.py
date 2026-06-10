from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

DatasetName = Literal["mnist", "cifar10", "cifar100"]

_DATASET_STATS: dict[DatasetName, tuple[tuple[float, ...], tuple[float, ...]]] = {
    "mnist": ((0.1307,), (0.3081,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}


class PackedDataLoader:
    """Iterate deterministic distributed shards as packed image batches."""

    def __init__(
        self,
        images: Tensor,
        targets: Tensor,
        *,
        batch_size: int,
        world_size: int,
        ranks: Sequence[int],
        base_seed: int,
        packed: bool = True,
        shuffle: bool = True,
        augment: bool = False,
        normalize: bool = True,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
        sampler_drop_last: bool = False,
        drop_last: bool = False,
    ) -> None:
        if images.ndim != 4:
            raise ValueError(f"images must have shape [N, C, H, W], got {tuple(images.shape)}")
        if images.shape[0] == 0:
            raise ValueError("images must not be empty")
        if targets.ndim != 1 or targets.shape[0] != images.shape[0]:
            raise ValueError("targets must have shape [N] and match the number of images")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {world_size}")
        if not ranks:
            raise ValueError("ranks must not be empty")
        if len(set(ranks)) != len(ranks):
            raise ValueError("ranks must be unique")
        if any(rank < 0 or rank >= world_size for rank in ranks):
            raise ValueError(f"all ranks must be in [0, {world_size})")
        if not packed and len(ranks) != 1:
            raise ValueError("unpacked mode requires exactly one rank")
        if augment and images.shape[-2:] != (32, 32):
            raise ValueError("built-in augmentation is supported only for 32x32 CIFAR images")
        if normalize and (mean is None or std is None):
            raise ValueError("mean and std are required when normalize=True")
        if mean is not None and len(mean) != images.shape[1]:
            raise ValueError("mean length must match the image channel count")
        if std is not None and len(std) != images.shape[1]:
            raise ValueError("std length must match the image channel count")
        if not images.is_floating_point():
            raise ValueError("images must be floating point values in [0, 1]")

        self.images = images
        self.targets = targets.to(device=images.device, dtype=torch.long)
        self.batch_size = batch_size
        self.world_size = world_size
        self.ranks = tuple(ranks)
        self.base_seed = base_seed
        self.packed = packed
        self.shuffle = shuffle
        self.augment = augment
        self.normalize = normalize
        self.sampler_drop_last = sampler_drop_last
        self.drop_last = drop_last
        self.epoch = 0
        self.num_samples = self._num_samples()
        self.total_size = self.num_samples * world_size
        self._mean = self._stat_tensor(mean)
        self._std = self._stat_tensor(std)

    @property
    def device(self) -> torch.device:
        return self.images.device

    def _stat_tensor(self, values: Sequence[float] | None) -> Tensor | None:
        if values is None:
            return None
        return torch.tensor(values, device=self.device, dtype=self.images.dtype).view(1, -1, 1, 1)

    def _num_samples(self) -> int:
        dataset_size = self.images.shape[0]
        if self.sampler_drop_last and dataset_size % self.world_size != 0:
            return math.ceil((dataset_size - self.world_size) / self.world_size)
        return math.ceil(dataset_size / self.world_size)

    def __len__(self) -> int:
        if self.drop_last:
            return self.num_samples // self.batch_size
        return math.ceil(self.num_samples / self.batch_size)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used by deterministic sampling and augmentation."""

        self.epoch = epoch

    def _distributed_indices(self) -> Tensor:
        dataset_size = self.images.shape[0]
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.base_seed + self.epoch)
            indices = torch.randperm(dataset_size, generator=generator)
        else:
            indices = torch.arange(dataset_size)

        if not self.sampler_drop_last:
            padding_size = self.total_size - dataset_size
            if padding_size <= dataset_size:
                indices = torch.cat((indices, indices[:padding_size]))
            else:
                indices = torch.cat((indices, indices.repeat(math.ceil(padding_size / dataset_size))[:padding_size]))
        else:
            indices = indices[: self.total_size]
        return indices

    def _augment_rank_batch(self, images: Tensor, rank: int, batch_start: int) -> Tensor:
        if not self.augment:
            return images

        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.base_seed + self.epoch * 1_000_003 + rank * 10_000_019 + batch_start)
        batch_size, _, height, width = images.shape
        offsets = torch.randint(0, 9, (batch_size, 2), device=self.device, generator=generator)
        padded = F.pad(images, (4, 4, 4, 4))
        rows = offsets[:, 0, None] + torch.arange(height, device=self.device)[None, :]
        cropped = padded.gather(2, rows[:, None, :, None].expand(-1, images.shape[1], -1, padded.shape[3]))
        columns = offsets[:, 1, None] + torch.arange(width, device=self.device)[None, :]
        cropped = cropped.gather(3, columns[:, None, None, :].expand(-1, images.shape[1], height, -1))
        flip = torch.rand(batch_size, device=self.device, generator=generator) < 0.5
        return torch.where(flip[:, None, None, None], cropped.flip(-1), cropped)

    def _prepare_rank_batch(self, indices: Tensor, rank: int, batch_start: int) -> tuple[Tensor, Tensor]:
        device_indices = indices.to(self.device)
        images = self._augment_rank_batch(self.images[device_indices], rank, batch_start)
        if self.normalize:
            assert self._mean is not None and self._std is not None
            images = (images - self._mean) / self._std
        return images, self.targets[device_indices]

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        indices = self._distributed_indices()
        rank_indices = [indices[rank : self.total_size : self.world_size] for rank in self.ranks]
        stop = self.num_samples if not self.drop_last else len(self) * self.batch_size

        for start in range(0, stop, self.batch_size):
            batches = [
                self._prepare_rank_batch(local_indices[start : start + self.batch_size], rank, start)
                for rank, local_indices in zip(self.ranks, rank_indices, strict=True)
            ]
            if self.packed:
                images = torch.stack([batch[0] for batch in batches], dim=1)
                images = images.flatten(1, 2).contiguous(memory_format=torch.channels_last)
                targets = torch.stack([batch[1] for batch in batches], dim=1)
            else:
                images, targets = batches[0]
                images = images.contiguous(memory_format=torch.channels_last)
            yield images, targets


def create_dataloader(
    dataset: DatasetName,
    *,
    root: str | Path,
    batch_size: int,
    world_size: int,
    ranks: Sequence[int],
    base_seed: int,
    train: bool = True,
    packed: bool = True,
    shuffle: bool | None = None,
    augment: bool | None = None,
    normalize: bool = True,
    device: torch.device | str | None = None,
    download: bool = True,
    sampler_drop_last: bool = False,
    drop_last: bool = False,
) -> PackedDataLoader:
    """Download a supported dataset and create a GPU-resident packed loader."""

    if dataset not in _DATASET_STATS:
        raise ValueError(f"unsupported dataset {dataset!r}; expected one of {tuple(_DATASET_STATS)}")
    if augment is True and not train:
        raise ValueError("augmentation is not supported for test splits")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    try:
        from torchvision import datasets
    except ImportError as error:
        raise ImportError("create_dataloader requires torchvision") from error

    dataset_types = {
        "mnist": datasets.MNIST,
        "cifar10": datasets.CIFAR10,
        "cifar100": datasets.CIFAR100,
    }
    source = dataset_types[dataset](root=str(root), train=train, download=download)
    images = torch.as_tensor(source.data)
    if dataset == "mnist":
        images = images.unsqueeze(1)
    else:
        images = images.permute(0, 3, 1, 2)
    images = images.to(device=device, dtype=torch.float32).div_(255)
    targets = torch.as_tensor(source.targets, dtype=torch.long, device=device)
    mean, std = _DATASET_STATS[dataset]

    return PackedDataLoader(
        images,
        targets,
        batch_size=batch_size,
        world_size=world_size,
        ranks=ranks,
        base_seed=base_seed,
        packed=packed,
        shuffle=train if shuffle is None else shuffle,
        augment=(train and dataset != "mnist") if augment is None else augment,
        normalize=normalize,
        mean=mean,
        std=std,
        sampler_drop_last=sampler_drop_last,
        drop_last=drop_last,
    )
