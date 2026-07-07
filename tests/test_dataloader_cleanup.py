from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
