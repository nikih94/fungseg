from __future__ import annotations

from typing import Any

from torch import nn

from src.models.norms import ChannelLayerNorm2d


def _normalize_decoder_attention_type(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return normalized or None


def _normalize_decoder_normalization(value: Any) -> bool | str | dict[str, Any]:
    if value is None:
        return "batchnorm"
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        normalized = dict(value)
        norm_type = str(normalized.get("type", "")).strip().lower()
        if norm_type == "instancenorm":
            normalized.setdefault("affine", True)
        return normalized

    normalized = str(value).strip().lower()
    aliases = {
        "batchnorm": "batchnorm",
        "batch_norm": "batchnorm",
        "bn": "batchnorm",
        "instancenorm": "instancenorm",
        "instance_norm": "instancenorm",
        "in": "instancenorm",
        "layernorm": "layernorm",
        "layer_norm": "layernorm",
        "ln": "layernorm",
        "identity": "identity",
        "none": "identity",
        "false": "identity",
    }
    if normalized not in aliases:
        raise ValueError(
            "Unsupported decoder_normalization: "
            f"{value}. Expected batchnorm, instancenorm, layernorm, or identity."
        )
    resolved = aliases[normalized]
    if resolved == "instancenorm":
        return {"type": "instancenorm", "affine": True}
    return resolved


def _replace_decoder_layer_norms(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.LayerNorm):
            normalized_shape = child.normalized_shape
            if not normalized_shape:
                raise ValueError("LayerNorm normalized_shape must be defined for decoder replacement.")
            if len(normalized_shape) != 1:
                raise ValueError(
                    "Only channel-wise LayerNorm is supported for decoder normalization."
                )
            setattr(
                module,
                name,
                ChannelLayerNorm2d(
                    num_channels=int(normalized_shape[0]),
                    eps=child.eps,
                    elementwise_affine=child.elementwise_affine,
                ),
            )
            continue
        _replace_decoder_layer_norms(child)


def build_model(config: dict[str, Any]):
    model_name = config["name"].lower()
    encoder_name = config.get("encoder_name", "resnet18")

    if model_name in {
        "unetplusplus",
        "unetplusplus_resnet18",
        "unetplusplus_resnet34",
        "unetplusplus_resnet50",
    }:
        import segmentation_models_pytorch as smp

        if model_name.startswith("unetplusplus_resnet"):
            encoder_name = model_name.replace("unetplusplus_", "")

        decoder_normalization = _normalize_decoder_normalization(
            config.get("decoder_normalization", "batchnorm")
        )
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=config.get("encoder_weights", "imagenet"),
            in_channels=config.get("in_channels", 3),
            classes=config.get("num_classes", 1),
            decoder_use_norm=decoder_normalization,
            decoder_channels=tuple(config.get("decoder_channels", [512, 256, 128, 64, 32])),
            decoder_attention_type=_normalize_decoder_attention_type(
                config.get("decoder_attention_type")
            ),
            activation=None,
        )
        if decoder_normalization == "layernorm" or (
            isinstance(decoder_normalization, dict)
            and str(decoder_normalization.get("type", "")).strip().lower() == "layernorm"
        ):
            _replace_decoder_layer_norms(model.decoder)
        return model

    if model_name == "deeplabv3_resnet50":
        import torchvision.models.segmentation as tv_segmentation

        weights = None
        weights_backbone = tv_segmentation.DeepLabV3_ResNet50_Weights.DEFAULT if config.get(
            "encoder_weights"
        ) == "imagenet" else None
        return tv_segmentation.deeplabv3_resnet50(
            weights=weights,
            weights_backbone=weights_backbone,
            num_classes=config.get("num_classes", 1),
        )

    if model_name == "fcn_resnet50":
        import torchvision.models.segmentation as tv_segmentation

        weights = None
        weights_backbone = tv_segmentation.FCN_ResNet50_Weights.DEFAULT if config.get(
            "encoder_weights"
        ) == "imagenet" else None
        return tv_segmentation.fcn_resnet50(
            weights=weights,
            weights_backbone=weights_backbone,
            num_classes=config.get("num_classes", 1),
        )

    raise ValueError(f"Unsupported model name: {config['name']}")
