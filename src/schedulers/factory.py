from __future__ import annotations

from typing import Any

from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR


def _resolve_min_lrs(optimizer, configured_min_lrs: dict[str, Any]) -> list[float]:
    group_names = [group.get("group_name") for group in optimizer.param_groups]
    if any(group_name is None for group_name in group_names):
        raise ValueError(
            "Named scheduler.min_lr values require named optimizer parameter groups."
        )
    if len(group_names) != len(set(group_names)):
        raise ValueError("Optimizer parameter group names must be unique.")

    expected_names = set(group_names)
    configured_names = set(configured_min_lrs)
    if configured_names != expected_names:
        raise ValueError(
            "scheduler.min_lr group names must match optimizer groups: "
            f"expected {sorted(expected_names)}, got {sorted(configured_names)}."
        )
    return [float(configured_min_lrs[group_name]) for group_name in group_names]


def build_scheduler(optimizer, config: dict[str, Any]):
    scheduler_name = config["name"].lower()
    kwargs = {key: value for key, value in config.items() if key not in {"name", "monitor"}}

    if scheduler_name in {"none", "null"}:
        return None
    if scheduler_name == "reduce_on_plateau":
        if isinstance(kwargs.get("min_lr"), dict):
            kwargs["min_lr"] = _resolve_min_lrs(optimizer, kwargs["min_lr"])
        return ReduceLROnPlateau(optimizer, **kwargs)
    if scheduler_name == "step":
        return StepLR(optimizer, **kwargs)
    if scheduler_name == "cosine":
        return CosineAnnealingLR(optimizer, **kwargs)

    raise ValueError(f"Unsupported scheduler name: {config['name']}")

