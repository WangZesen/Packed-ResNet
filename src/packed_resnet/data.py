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
    """Iterate selected deterministic distributed shards as image batches.

    The loader simulates multiple distributed workers in one process. Sampling
    matches :class:`torch.utils.data.DistributedSampler`: all ranks share the
    permutation generated from ``base_seed + epoch``, then each selected rank
    receives its strided shard. Call :meth:`set_epoch` before each epoch.

    Images and all epoch preprocessing remain on the input tensor's device. The
    complete selected-rank epoch is gathered, augmented, normalized, and packed
    before inexpensive batch views are yielded.

    Args:
        images: Floating-point source images shaped ``(N, C, H, W)`` with
            values in ``[0, 1]``.
        targets: Class targets shaped ``(N,)``.
        local_batch_size: Number of samples yielded for each selected rank in
            one batch.
        world_size: Total number of simulated distributed workers.
        ranks: Unique simulated worker ranks to include, in packed output
            order. Every rank must be in ``[0, world_size)``.
        base_seed: Base seed for distributed shuffling and rank-stable
            augmentation.
        packed: If ``True``, combine selected rank images along the channel
            dimension. If ``False``, exactly one rank must be selected.
        channels_last: If ``True``, return channels-last contiguous image
            tensors. Otherwise return standard contiguous NCHW tensors.
        shuffle: If ``True``, deterministically shuffle before distributed
            sharding.
        augment: If ``True``, apply deterministic padded random crops. CIFAR
            images additionally receive random horizontal flips.
        normalize: If ``True``, normalize images using ``mean`` and ``std``.
        mean: Per-channel normalization means. Required when ``normalize`` is
            ``True``.
        std: Per-channel normalization standard deviations. Required when
            ``normalize`` is ``True``.
        sampler_drop_last: Match ``DistributedSampler(drop_last=True)`` by
            dropping the tail needed to make shards evenly divisible.
        drop_last: Drop each rank's final incomplete local batch.

    Yields:
        ``(images, targets)`` batches. Packed images have shape
        ``(B, K * C, H, W)`` and targets have shape ``(B, K)``. Unpacked images
        have shape ``(B, C, H, W)`` and targets have shape ``(B,)``.

    Raises:
        ValueError: If tensor shapes, ranks, batch settings, augmentation
            settings, or normalization statistics are invalid.
    """

    def __init__(
        self,
        images: Tensor,
        targets: Tensor,
        *,
        local_batch_size: int,
        world_size: int,
        ranks: Sequence[int],
        base_seed: int,
        packed: bool = True,
        channels_last: bool = True,
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
        if local_batch_size < 1:
            raise ValueError(f"local_batch_size must be >= 1, got {local_batch_size}")
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
        if augment and tuple(images.shape[1:]) not in ((1, 28, 28), (3, 32, 32)):
            raise ValueError("built-in augmentation supports only MNIST [N, 1, 28, 28] and CIFAR [N, 3, 32, 32]")
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
        self.local_batch_size = local_batch_size
        self.world_size = world_size
        self.ranks = tuple(ranks)
        self.base_seed = base_seed
        self.packed = packed
        self.channels_last = channels_last
        self.shuffle = shuffle
        self.augment = augment
        self._horizontal_flip = images.shape[1:] == (3, 32, 32)
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
        """Return the device holding the source dataset and yielded batches."""

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
        """Return the number of local batches yielded per epoch."""

        if self.drop_last:
            return self.num_samples // self.local_batch_size
        return math.ceil(self.num_samples / self.local_batch_size)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used by deterministic sampling and augmentation.

        Args:
            epoch: Epoch number mixed into sampling and augmentation seeds.
        """

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

    def _augmentation_parameters(self) -> tuple[Tensor, Tensor] | None:
        if not self.augment:
            return None

        offsets: list[Tensor] = []
        flips: list[Tensor] = []
        for rank in self.ranks:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.base_seed + self.epoch * 1_000_003 + rank * 10_000_019)
            offsets.append(torch.randint(0, 9, (self.num_samples, 2), device=self.device, generator=generator))
            if self._horizontal_flip:
                flips.append(torch.rand(self.num_samples, device=self.device, generator=generator) < 0.5)
            else:
                flips.append(torch.zeros(self.num_samples, device=self.device, dtype=torch.bool))
        return torch.stack(offsets, dim=1), torch.stack(flips, dim=1)

    def _augment_batch(self, images: Tensor, offsets: Tensor, flip: Tensor) -> Tensor:
        num_images, _, height, width = images.shape
        padded = F.pad(images, (4, 4, 4, 4))
        rows = offsets[:, 0, None] + torch.arange(height, device=self.device)[None, :]
        cropped = padded.gather(2, rows[:, None, :, None].expand(-1, images.shape[1], -1, padded.shape[3]))
        column_order = torch.arange(width, device=self.device).expand(num_images, -1)
        if self._horizontal_flip:
            column_order = torch.where(flip[:, None], width - 1 - column_order, column_order)
        columns = offsets[:, 1, None] + column_order
        return cropped.gather(3, columns[:, None, None, :].expand(-1, images.shape[1], height, -1))

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        """Materialize the selected-rank epoch and iterate over batch views."""

        indices = self._distributed_indices()
        rank_indices = torch.stack(
            [indices[rank : self.total_size : self.world_size] for rank in self.ranks],
            dim=1,
        )
        augmentation_parameters = self._augmentation_parameters()
        stop = self.num_samples if not self.drop_last else len(self) * self.local_batch_size

        # Materialize preprocessing once so iteration only slices batch views.
        epoch_indices = rank_indices[:stop].to(self.device)
        epoch_images = self.images[epoch_indices].flatten(0, 1)
        epoch_targets = self.targets[epoch_indices]
        if augmentation_parameters is not None:
            offsets, flips = augmentation_parameters
            epoch_images = self._augment_batch(
                epoch_images,
                offsets[:stop].flatten(0, 1),
                flips[:stop].flatten(),
            )
        if self.normalize:
            assert self._mean is not None and self._std is not None
            epoch_images = (epoch_images - self._mean) / self._std
        if self.packed:
            epoch_images = epoch_images.unflatten(0, (stop, len(self.ranks)))
            epoch_images = epoch_images.flatten(1, 2)
        else:
            epoch_targets = epoch_targets[:, 0]
        if self.channels_last:
            epoch_images = epoch_images.contiguous(memory_format=torch.channels_last)
        else:
            epoch_images = epoch_images.contiguous()

        for start in range(0, stop, self.local_batch_size):
            end = start + self.local_batch_size
            yield epoch_images[start:end], epoch_targets[start:end]


def create_dataloader(
    dataset: DatasetName,
    *,
    root: str | Path = "./data",
    local_batch_size: int,
    world_size: int,
    ranks: Sequence[int],
    base_seed: int,
    train: bool = True,
    packed: bool = True,
    channels_last: bool = True,
    shuffle: bool | None = None,
    augment: bool | None = None,
    device: torch.device | str | None = None,
    sampler_drop_last: bool = False,
    drop_last: bool = False,
) -> PackedDataLoader:
    """Create a normalized packed loader for MNIST, CIFAR10, or CIFAR100.

    Missing torchvision data is downloaded under ``root``. The complete split
    is converted to ``float32``, normalized with standard dataset statistics,
    and stored on ``device``. CUDA is selected automatically when available.

    Args:
        dataset: Dataset name: ``"mnist"``, ``"cifar10"``, or ``"cifar100"``.
        root: Dataset download and storage directory. Default: ``"./data"``.
        local_batch_size: Number of samples per selected simulated rank in one
            yielded batch.
        world_size: Total number of simulated distributed workers.
        ranks: Unique worker ranks to include, in packed output order.
        base_seed: Base seed for shuffling and deterministic augmentation.
        train: If ``True``, load the training split. Otherwise load the test
            split.
        packed: If ``True``, pack selected rank images along the channel
            dimension. If ``False``, ``ranks`` must contain exactly one rank.
        channels_last: If ``True``, yield channels-last contiguous images.
            Otherwise yield standard contiguous NCHW images.
        shuffle: Override split-dependent shuffling. By default, training data
            is shuffled and test data is not.
        augment: Override split-dependent augmentation. By default, CIFAR
            training data uses random crops and horizontal flips, while MNIST
            and test splits are not augmented. Explicit MNIST augmentation
            applies random crops without flips.
        device: Dataset storage and output device. By default, use CUDA when
            available and CPU otherwise.
        sampler_drop_last: Drop distributed-sampler tail samples instead of
            padding shards to equal length.
        drop_last: Drop each rank's final incomplete local batch.

    Returns:
        A :class:`PackedDataLoader` holding the requested normalized split.

    Raises:
        ValueError: If the dataset name or split/augmentation combination is
            unsupported.
        ImportError: If torchvision is unavailable.
    """

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
    source = dataset_types[dataset](root=str(root), train=train, download=True)
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
        local_batch_size=local_batch_size,
        world_size=world_size,
        ranks=ranks,
        base_seed=base_seed,
        packed=packed,
        channels_last=channels_last,
        shuffle=train if shuffle is None else shuffle,
        augment=(train and dataset != "mnist") if augment is None else augment,
        normalize=True,
        mean=mean,
        std=std,
        sampler_drop_last=sampler_drop_last,
        drop_last=drop_last,
    )
