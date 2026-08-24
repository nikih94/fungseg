from __future__ import annotations

import unittest

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from src.data.dataset import get_train_transforms
from src.engine.trainer import shutdown_dataloader
from src.train import TORCH_SHARING_STRATEGY, configure_torch_multiprocessing, make_loader


class _FakeIterator:
    def __init__(self) -> None:
        self.shutdown_calls = 0

    def _shutdown_workers(self) -> None:
        self.shutdown_calls += 1


class _FakeLoader:
    def __init__(self) -> None:
        self._iterator = _FakeIterator()


class _AugmentedWorkerDataset(Dataset):
    def __init__(self) -> None:
        y, x = np.mgrid[:48, :48]
        self.image = np.stack(
            [x.astype(np.uint8), y.astype(np.uint8), (x + y).astype(np.uint8)],
            axis=-1,
        )
        self.transforms = A.Compose(
            [
                A.Affine(
                    translate_percent={"x": (-0.35, 0.35), "y": (-0.35, 0.35)},
                    rotate=(-35, 35),
                    p=1.0,
                ),
                ToTensorV2(),
            ]
        )

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        worker_info = torch.utils.data.get_worker_info()
        transformed = self.transforms(image=self.image)
        return {
            "image": transformed["image"],
            "worker_id": -1 if worker_info is None else worker_info.id,
        }


class DataLoaderCleanupTests(unittest.TestCase):
    def test_shutdown_dataloader_stops_cached_persistent_iterator(self) -> None:
        loader = _FakeLoader()
        iterator = loader._iterator

        shutdown_dataloader(loader)

        self.assertEqual(iterator.shutdown_calls, 1)
        self.assertIsNone(loader._iterator)

    def test_shutdown_dataloader_accepts_none(self) -> None:
        shutdown_dataloader(None)

    def test_training_uses_filesystem_tensor_sharing_for_workers(self) -> None:
        self.assertEqual(configure_torch_multiprocessing(), TORCH_SHARING_STRATEGY)

        loader = make_loader(
            dataset=[],
            batch_size=1,
            num_workers=1,
            pin_memory=False,
            shuffle=False,
            persistent_workers=False,
            prefetch_factor=2,
        )

        self.assertIsNotNone(loader.worker_init_fn)

    def test_seeded_train_transforms_repeat_without_workers(self) -> None:
        image = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:24, 10:22] = 1
        first_transform = get_train_transforms(seed=42)
        repeat_transform = get_train_transforms(seed=42)

        first = first_transform(image=image, mask=mask)
        repeat = repeat_transform(image=image, mask=mask)

        self.assertTrue(torch.equal(first["image"], repeat["image"]))
        self.assertTrue(torch.equal(first["mask"], repeat["mask"]))

    def test_workers_receive_distinct_reproducible_augmentation_streams(self) -> None:
        def collect_first_two() -> list[tuple[int, torch.Tensor]]:
            torch.manual_seed(123)
            loader = make_loader(
                dataset=_AugmentedWorkerDataset(),
                batch_size=1,
                num_workers=2,
                pin_memory=False,
                shuffle=False,
                persistent_workers=False,
                prefetch_factor=2,
            )
            samples: list[tuple[int, torch.Tensor]] = []
            try:
                for batch in loader:
                    samples.append((int(batch["worker_id"].item()), batch["image"].clone()))
                    if len(samples) == 2:
                        break
            finally:
                shutdown_dataloader(loader)
            return samples

        first = collect_first_two()
        repeat = collect_first_two()

        self.assertEqual([worker_id for worker_id, _ in first], [0, 1])
        self.assertFalse(torch.equal(first[0][1], first[1][1]))
        self.assertEqual([worker_id for worker_id, _ in first], [worker_id for worker_id, _ in repeat])
        for (_, first_image), (_, repeat_image) in zip(first, repeat):
            self.assertTrue(torch.equal(first_image, repeat_image))


if __name__ == "__main__":
    unittest.main()
