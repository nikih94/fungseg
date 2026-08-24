from __future__ import annotations

from typing import Any

from torch import nn, optim


def _split_parameter_groups(model: nn.Module, config: dict[str, Any]) -> list[dict[str, Any]]:
    if not hasattr(model, "encoder"):
        raise ValueError(
            "Separate encoder_lr and decoder_lr require a model with an 'encoder' module."
        )

    encoder_parameters = [
        parameter for parameter in model.encoder.parameters() if parameter.requires_grad
    ]
    encoder_parameter_ids = {id(parameter) for parameter in encoder_parameters}
    decoder_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in encoder_parameter_ids
    ]
    if not encoder_parameters:
        raise ValueError("The model encoder has no trainable parameters.")
    if not decoder_parameters:
        raise ValueError("The model decoder group has no trainable parameters.")

    return [
        {
            "params": encoder_parameters,
            "lr": float(config["encoder_lr"]),
            "group_name": "encoder",
        },
        {
            "params": decoder_parameters,
            "lr": float(config["decoder_lr"]),
            "group_name": "decoder",
        },
    ]


def build_optimizer(model_or_parameters, config: dict[str, Any]):
    optimizer_name = config["name"].lower()
    has_encoder_lr = "encoder_lr" in config
    has_decoder_lr = "decoder_lr" in config
    if has_encoder_lr != has_decoder_lr:
        raise ValueError(
            "optimizer.encoder_lr and optimizer.decoder_lr must be configured together."
        )

    split_learning_rates = has_encoder_lr and has_decoder_lr
    excluded_keys = {"name", "encoder_lr", "decoder_lr"}
    if split_learning_rates:
        if not isinstance(model_or_parameters, nn.Module):
            raise ValueError(
                "Separate encoder_lr and decoder_lr require passing the model to build_optimizer."
            )
        parameters = _split_parameter_groups(model_or_parameters, config)
        excluded_keys.add("lr")
    else:
        parameters = (
            model_or_parameters.parameters()
            if isinstance(model_or_parameters, nn.Module)
            else model_or_parameters
        )

    kwargs = {key: value for key, value in config.items() if key not in excluded_keys}

    if optimizer_name == "adam":
        return optim.Adam(parameters, **kwargs)
    if optimizer_name == "adamw":
        return optim.AdamW(parameters, **kwargs)
    if optimizer_name == "sgd":
        return optim.SGD(parameters, **kwargs)

    raise ValueError(f"Unsupported optimizer name: {config['name']}")

