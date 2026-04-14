from __future__ import annotations

from typing import Any

from torch import optim


def build_optimizer(parameters, config: dict[str, Any]):
    optimizer_name = config["name"].lower()
    kwargs = {key: value for key, value in config.items() if key != "name"}

    if optimizer_name == "adam":
        return optim.Adam(parameters, **kwargs)
    if optimizer_name == "adamw":
        return optim.AdamW(parameters, **kwargs)
    if optimizer_name == "sgd":
        return optim.SGD(parameters, **kwargs)

    raise ValueError(f"Unsupported optimizer name: {config['name']}")

