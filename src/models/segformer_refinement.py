from __future__ import annotations

from collections.abc import Sequence

import torch
from segmentation_models_pytorch.base.initialization import (
    initialize_decoder,
    initialize_head,
)
from segmentation_models_pytorch.decoders.segformer.decoder import SegformerDecoder
from segmentation_models_pytorch.encoders import get_encoder
from torch import nn
from torch.nn import functional as F


SUPPORTED_MIT_ENCODERS = {"mit_b1", "mit_b2"}
MIT_B1_B2_FEATURE_CHANNELS = (64, 128, 320, 512)


def _channel_pair(name: str, values: Sequence[int]) -> tuple[int, int]:
    error_message = f"{name} must contain exactly two positive channel counts."
    try:
        if isinstance(values, (str, bytes)):
            raise ValueError
        channels = tuple(int(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError(error_message) from error
    if len(channels) != 2 or any(channel <= 0 for channel in channels):
        raise ValueError(error_message)
    return channels


def _conv_norm_gelu(
    in_channels: int,
    out_channels: int,
    *,
    stride: int = 1,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.GELU(),
    )


def _refinement_block(
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
) -> nn.Sequential:
    return nn.Sequential(
        *_conv_norm_gelu(in_channels, hidden_channels),
        *_conv_norm_gelu(hidden_channels, out_channels),
    )


class SegformerMitFullResolutionRefinement(nn.Module):
    """MiT-B1/B2 SegFormer with shallow half/full-resolution refinement."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        num_classes: int = 3,
        encoder_name: str = "mit_b1",
        encoder_weights: str | None = "imagenet",
        encoder_depth: int = 5,
        decoder_segmentation_channels: int = 256,
        shallow_channels: Sequence[int] = (16, 32),
        refine_half_channels: Sequence[int] = (128, 64),
        refine_full_channels: Sequence[int] = (32, 32),
    ) -> None:
        super().__init__()

        if encoder_name not in SUPPORTED_MIT_ENCODERS:
            raise ValueError(
                "SegformerMitFullResolutionRefinement requires encoder_name "
                "to be 'mit_b1' or 'mit_b2'."
            )
        if encoder_depth != 5:
            raise ValueError(
                "SegformerMitFullResolutionRefinement requires encoder_depth=5 "
                "to expose all four MiT stages."
            )
        if in_channels <= 0 or num_classes <= 0 or decoder_segmentation_channels <= 0:
            raise ValueError(
                "in_channels, num_classes, and decoder_segmentation_channels must be positive."
            )

        f0_channels, f1_channels = _channel_pair("shallow_channels", shallow_channels)
        half_hidden, half_out = _channel_pair(
            "refine_half_channels", refine_half_channels
        )
        full_hidden, full_out = _channel_pair(
            "refine_full_channels", refine_full_channels
        )

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        actual_encoder_channels = tuple(
            int(value) for value in self.encoder.out_channels[-4:]
        )
        if actual_encoder_channels != MIT_B1_B2_FEATURE_CHANNELS:
            raise ValueError(
                f"Unexpected {encoder_name} feature channels: "
                f"{actual_encoder_channels}; expected {MIT_B1_B2_FEATURE_CHANNELS}."
            )

        self.decoder = SegformerDecoder(
            encoder_channels=list(self.encoder.out_channels),
            encoder_depth=encoder_depth,
            segmentation_channels=decoder_segmentation_channels,
        )
        self.shallow_full = _conv_norm_gelu(in_channels, f0_channels)
        self.shallow_half = _conv_norm_gelu(
            f0_channels,
            f1_channels,
            stride=2,
        )
        self.refine_half = _refinement_block(
            decoder_segmentation_channels + f1_channels,
            half_hidden,
            half_out,
        )
        self.refine_full = _refinement_block(
            half_out + f0_channels,
            full_hidden,
            full_out,
        )
        self.segmentation_head = nn.Conv2d(full_out, num_classes, kernel_size=1)

        initialize_decoder(self.decoder)
        initialize_decoder(self.shallow_full)
        initialize_decoder(self.shallow_half)
        initialize_decoder(self.refine_half)
        initialize_decoder(self.refine_full)
        initialize_head(self.segmentation_head)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        full_resolution = self.shallow_full(inputs)
        half_resolution = self.shallow_half(full_resolution)

        decoder_features = self.decoder(self.encoder(inputs))
        decoder_features = F.interpolate(
            decoder_features,
            size=half_resolution.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        refined_half = self.refine_half(
            torch.cat((decoder_features, half_resolution), dim=1)
        )

        refined_half = F.interpolate(
            refined_half,
            size=full_resolution.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        refined_full = self.refine_full(
            torch.cat((refined_half, full_resolution), dim=1)
        )
        return self.segmentation_head(refined_full)


# Backwards-compatible import name retained for existing callers.
SegformerMitB1FullResolutionRefinement = SegformerMitFullResolutionRefinement
