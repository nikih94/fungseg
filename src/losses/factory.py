from __future__ import annotations

from typing import Any

from torch import nn

from src.losses.combined import (
    BCEDiceLoss,
    BCEDiceSoftCLDiceLoss,
    CLDiceLoss,
    SoftCLDiceLoss,
    TverskyLoss,
    TverskySoftCLDiceLoss,
)


def build_loss(config: dict[str, Any]):
    loss_name = config["name"].lower()
    if loss_name in {"bce_with_logits", "bce", "binary_cross_entropy_with_logits"}:
        return nn.BCEWithLogitsLoss()
    if loss_name == "bce_dice":
        return BCEDiceLoss(
            bce_weight=float(config.get("bce_weight", 0.5)),
            dice_weight=float(config.get("dice_weight", 0.5)),
        )
    if loss_name in {"bce_dice_cldice", "bce_dice_soft_cldice", "bcedicecldice"}:
        return BCEDiceSoftCLDiceLoss(
            bce_weight=float(config.get("bce_weight", 0.3)),
            dice_weight=float(config.get("dice_weight", 0.6)),
            soft_cldice_weight=float(config.get("soft_cldice_weight", 0.1)),
            iterations=int(config.get("iterations", 5)),
            smooth=float(config.get("smooth", 1e-6)),
            cldice_smooth=float(config.get("cldice_smooth", 1.0)),
        )
    if loss_name == "tversky":
        return TverskyLoss(
            alpha=float(config.get("alpha", 0.3)),
            beta=float(config.get("beta", 0.7)),
            smooth=float(config.get("smooth", 1e-6)),
        )
    if loss_name == "cldice":
        return CLDiceLoss(
            threshold=float(config.get("threshold", 0.5)),
            iterations=int(config.get("iterations", 3)),
            smooth=float(config.get("cldice_smooth", config.get("smooth", 1.0))),
        )
    if loss_name in {"soft_cldice", "softcldice"}:
        return SoftCLDiceLoss(
            iterations=int(config.get("iterations", 3)),
            smooth=float(config.get("cldice_smooth", config.get("smooth", 1.0))),
        )
    if loss_name in {"tversky_soft_cldice", "tversky_softcldice"}:
        return TverskySoftCLDiceLoss(
            alpha=float(config.get("alpha", 0.3)),
            beta=float(config.get("beta", 0.7)),
            tversky_weight=float(config.get("tversky_weight", 0.7)),
            soft_cldice_weight=float(config.get("soft_cldice_weight", 0.3)),
            iterations=int(config.get("iterations", 3)),
            smooth=float(config.get("smooth", 1e-6)),
            cldice_smooth=float(config.get("cldice_smooth", 1.0)),
        )
    raise ValueError(f"Unsupported loss name: {config['name']}")
