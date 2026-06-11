from __future__ import annotations

import unittest

import torch

from src.models.factory import build_model
from src.models.wrappers import extract_logits
from src.utils.config import load_config


class ModelFactoryTests(unittest.TestCase):
    def test_builds_segformer_mit_b3(self) -> None:
        config = load_config("config_segformer_mit_b3.yaml")
        config["model"]["encoder_weights"] = None

        model = build_model(config["model"])
        model.eval()

        with torch.no_grad():
            logits = extract_logits(model(torch.zeros(1, 3, 64, 64)))

        self.assertEqual(tuple(logits.shape), (1, 1, 64, 64))


if __name__ == "__main__":
    unittest.main()
