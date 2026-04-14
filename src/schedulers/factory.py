from __future__ import annotations

from typing import Any

from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR


def build_scheduler(optimizer, config: dict[str, Any]):
    scheduler_name = config["name"].lower()
    kwargs = {key: value for key, value in config.items() if key not in {"name", "monitor"}}

    if scheduler_name in {"none", "null"}:
        return None
    if scheduler_name == "reduce_on_plateau":
        return ReduceLROnPlateau(optimizer, **kwargs)
    if scheduler_name == "step":
        return StepLR(optimizer, **kwargs)
    if scheduler_name == "cosine":
        return CosineAnnealingLR(optimizer, **kwargs)

    raise ValueError(f"Unsupported scheduler name: {config['name']}")

