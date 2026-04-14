from __future__ import annotations

from typing import Any

from src.losses.combined import BCEDiceLoss


def build_loss(config: dict[str, Any]):
    loss_name = config["name"].lower()
    if loss_name == "bce_dice":
        return BCEDiceLoss(
            bce_weight=float(config.get("bce_weight", 0.5)),
            dice_weight=float(config.get("dice_weight", 0.5)),
        )
    raise ValueError(f"Unsupported loss name: {config['name']}")

