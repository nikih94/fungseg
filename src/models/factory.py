from __future__ import annotations

from typing import Any


def _normalize_decoder_attention_type(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return normalized or None


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

        return smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=config.get("encoder_weights", "imagenet"),
            in_channels=config.get("in_channels", 3),
            classes=config.get("num_classes", 1),
            decoder_channels=tuple(config.get("decoder_channels", [512, 256, 128, 64, 32])),
            decoder_attention_type=_normalize_decoder_attention_type(
                config.get("decoder_attention_type")
            ),
            activation=None,
        )

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
