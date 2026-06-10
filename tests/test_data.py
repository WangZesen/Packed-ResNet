from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DistributedSampler, TensorDataset

from packed_resnet import PackedDataLoader, create_dataloader


def _images(size: int, channels: int = 3, height: int = 4, width: int = 4) -> torch.Tensor:
    values = torch.arange(size, dtype=torch.float32).view(size, 1, 1, 1)
    return values.expand(-1, channels, height, width).clone()


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("sampler_drop_last", [False, True])
def test_indices_exactly_match_distributed_sampler(shuffle: bool, sampler_drop_last: bool) -> None:
    size = 17
    world_size = 4
    epoch = 3
    loader = PackedDataLoader(
        _images(size),
        torch.arange(size),
        batch_size=3,
        world_size=world_size,
        ranks=[3, 1],
        base_seed=19,
        shuffle=shuffle,
        normalize=False,
        sampler_drop_last=sampler_drop_last,
    )
    loader.set_epoch(epoch)

    actual = loader._distributed_indices()

    for rank in loader.ranks:
        sampler = DistributedSampler(
            TensorDataset(torch.arange(size)),
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=19,
            drop_last=sampler_drop_last,
        )
        sampler.set_epoch(epoch)
        assert actual[rank : loader.total_size : world_size].tolist() == list(sampler)


def test_packed_batches_follow_rank_order_and_are_channels_last() -> None:
    loader = PackedDataLoader(
        _images(12),
        torch.arange(12),
        batch_size=2,
        world_size=3,
        ranks=[2, 0],
        base_seed=0,
        shuffle=False,
        normalize=False,
    )

    images, targets = next(iter(loader))

    assert images.shape == (2, 6, 4, 4)
    assert images.is_contiguous(memory_format=torch.channels_last)
    assert targets.tolist() == [[2, 0], [5, 3]]
    torch.testing.assert_close(images[:, :3], _images(12)[torch.tensor([2, 5])])
    torch.testing.assert_close(images[:, 3:], _images(12)[torch.tensor([0, 3])])


def test_unpacked_batches_and_validation() -> None:
    loader = PackedDataLoader(
        _images(8),
        torch.arange(8),
        batch_size=2,
        world_size=2,
        ranks=[1],
        base_seed=0,
        packed=False,
        shuffle=False,
        normalize=False,
    )

    images, targets = next(iter(loader))

    assert images.shape == (2, 3, 4, 4)
    assert images.is_contiguous(memory_format=torch.channels_last)
    assert targets.tolist() == [1, 3]
    with pytest.raises(ValueError, match="exactly one rank"):
        PackedDataLoader(
            _images(8),
            torch.arange(8),
            batch_size=2,
            world_size=2,
            ranks=[0, 1],
            base_seed=0,
            packed=False,
            normalize=False,
        )


def test_drop_last_and_length() -> None:
    keep = PackedDataLoader(
        _images(10), torch.arange(10), batch_size=2, world_size=3, ranks=[0], base_seed=0, normalize=False
    )
    drop_batch = PackedDataLoader(
        _images(10),
        torch.arange(10),
        batch_size=3,
        world_size=3,
        ranks=[0],
        base_seed=0,
        normalize=False,
        drop_last=True,
    )
    drop_sampler = PackedDataLoader(
        _images(10),
        torch.arange(10),
        batch_size=2,
        world_size=3,
        ranks=[0],
        base_seed=0,
        normalize=False,
        sampler_drop_last=True,
    )

    assert keep.num_samples == 4
    assert len(keep) == 2
    assert len(drop_batch) == 1
    assert drop_sampler.num_samples == 3
    assert len(drop_sampler) == 2


def test_normalization() -> None:
    loader = PackedDataLoader(
        torch.full((4, 1, 2, 2), 0.5),
        torch.arange(4),
        batch_size=2,
        world_size=1,
        ranks=[0],
        base_seed=0,
        packed=False,
        shuffle=False,
        mean=(0.25,),
        std=(0.5,),
    )

    images, _ = next(iter(loader))

    torch.testing.assert_close(images, torch.full_like(images, 0.5))


def test_augmentation_is_deterministic_and_rank_stable() -> None:
    images = torch.arange(12 * 3 * 32 * 32, dtype=torch.float32).reshape(12, 3, 32, 32)
    common = dict(
        images=images,
        targets=torch.arange(12),
        batch_size=2,
        world_size=3,
        base_seed=7,
        shuffle=True,
        augment=True,
        normalize=False,
    )
    combined = PackedDataLoader(ranks=[2, 0], **common)
    rank_only = PackedDataLoader(ranks=[2], **common)

    combined_images, combined_targets = next(iter(combined))
    repeated_images, repeated_targets = next(iter(combined))
    rank_images, rank_targets = next(iter(rank_only))

    torch.testing.assert_close(combined_images, repeated_images)
    torch.testing.assert_close(combined_targets, repeated_targets)
    torch.testing.assert_close(combined_images[:, :3], rank_images)
    torch.testing.assert_close(combined_targets[:, 0], rank_targets[:, 0])

    combined.set_epoch(1)
    next_epoch_images, _ = next(iter(combined))
    assert not torch.equal(combined_images, next_epoch_images)


def test_augmentation_is_independent_of_batch_size() -> None:
    images = torch.arange(12 * 3 * 32 * 32, dtype=torch.float32).reshape(12, 3, 32, 32)
    common = dict(
        images=images,
        targets=torch.arange(12),
        world_size=2,
        ranks=[0, 1],
        base_seed=7,
        shuffle=True,
        augment=True,
        normalize=False,
    )
    small_batches = PackedDataLoader(batch_size=2, **common)
    large_batches = PackedDataLoader(batch_size=4, **common)

    small_images, small_targets = zip(*small_batches, strict=True)
    large_images, large_targets = zip(*large_batches, strict=True)

    torch.testing.assert_close(torch.cat(small_images), torch.cat(large_images))
    torch.testing.assert_close(torch.cat(small_targets), torch.cat(large_targets))
    assert all(batch.is_contiguous(memory_format=torch.channels_last) for batch in small_images)
    assert all(batch.is_contiguous(memory_format=torch.channels_last) for batch in large_images)


def test_mnist_augmentation_applies_random_crop_without_horizontal_flip() -> None:
    images = torch.arange(8 * 28 * 28, dtype=torch.float32).reshape(8, 1, 28, 28)
    loader = PackedDataLoader(
        images,
        torch.arange(8),
        batch_size=4,
        world_size=1,
        ranks=[0],
        base_seed=7,
        packed=False,
        shuffle=False,
        augment=True,
        normalize=False,
    )

    offsets, flips = loader._augmentation_parameters() or (None, None)
    first_images, _ = next(iter(loader))
    repeated_images, _ = next(iter(loader))

    assert offsets is not None and flips is not None
    assert torch.count_nonzero(flips) == 0
    assert first_images.shape == (4, 1, 28, 28)
    assert first_images.is_contiguous(memory_format=torch.channels_last)
    assert not torch.equal(first_images, images[:4])
    torch.testing.assert_close(first_images, repeated_images)


@pytest.mark.parametrize(
    ("dataset_name", "shape", "channels", "default_augment"),
    [
        ("mnist", (6, 28, 28), 1, False),
        ("cifar10", (6, 32, 32, 3), 3, True),
        ("cifar100", (6, 32, 32, 3), 3, True),
    ],
)
def test_factory_loads_supported_datasets(
    monkeypatch: pytest.MonkeyPatch,
    dataset_name: str,
    shape: tuple[int, ...],
    channels: int,
    default_augment: bool,
) -> None:
    class FakeDataset:
        def __init__(self, root: str, train: bool, download: bool) -> None:
            del root, train, download
            self.data = torch.zeros(shape, dtype=torch.uint8).numpy()
            self.targets = list(range(shape[0]))

    fake_datasets = SimpleNamespace(MNIST=FakeDataset, CIFAR10=FakeDataset, CIFAR100=FakeDataset)
    monkeypatch.setitem(sys.modules, "torchvision", SimpleNamespace(datasets=fake_datasets))

    loader = create_dataloader(
        dataset_name,
        root="unused",
        batch_size=2,
        world_size=2,
        ranks=[0, 1],
        base_seed=0,
        device="cpu",
    )

    assert loader.images.shape == (shape[0], channels, shape[-2], shape[-2])
    assert loader.images.dtype == torch.float32
    assert loader.device.type == "cpu"
    assert loader.augment is default_augment


def test_factory_rejects_test_augmentation_and_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="test splits"):
        create_dataloader(
            "cifar10",
            root="unused",
            batch_size=2,
            world_size=1,
            ranks=[0],
            base_seed=0,
            train=False,
            augment=True,
        )
    with pytest.raises(ValueError, match="ranks must be unique"):
        PackedDataLoader(
            _images(4), torch.arange(4), batch_size=1, world_size=2, ranks=[0, 0], base_seed=0, normalize=False
        )
    with pytest.raises(ValueError, match=r"\[0, 2\)"):
        PackedDataLoader(
            _images(4), torch.arange(4), batch_size=1, world_size=2, ranks=[2], base_seed=0, normalize=False
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_loader_keeps_and_returns_data_on_cuda() -> None:
    loader = PackedDataLoader(
        _images(4).cuda(),
        torch.arange(4),
        batch_size=2,
        world_size=1,
        ranks=[0],
        base_seed=0,
        normalize=False,
    )

    images, targets = next(iter(loader))

    assert loader.images.device.type == "cuda"
    assert images.device.type == "cuda"
    assert targets.device.type == "cuda"
