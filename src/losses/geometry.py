from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


BACKGROUND_CLASS = 0
LOCI_CLASS = 1
INOCULUM_CLASS = 2


@dataclass(frozen=True)
class GeometryWeightMapBuilder:
    center_multiplier: float = 2.0
    center_gamma: float = 2.0
    separator_enabled: bool = True
    separator_multiplier: float = 3.0
    separator_radius_multipliers: tuple[float, ...] = (0.5, 1.0, 1.5)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GeometryWeightMapBuilder":
        multipliers = tuple(
            float(value)
            for value in config.get("separator_radius_multipliers", [0.5, 1.0, 1.5])
        )
        builder = cls(
            center_multiplier=float(config.get("center_multiplier", 2.0)),
            center_gamma=float(config.get("center_gamma", 2.0)),
            separator_enabled=bool(config.get("separator_enabled", True)),
            separator_multiplier=float(config.get("separator_multiplier", 3.0)),
            separator_radius_multipliers=multipliers,
        )
        builder.validate()
        return builder

    def validate(self) -> None:
        if self.center_multiplier < 0.0:
            raise ValueError("geometry_aware_ce.center_multiplier must be non-negative.")
        if self.center_gamma <= 0.0:
            raise ValueError("geometry_aware_ce.center_gamma must be positive.")
        if self.separator_multiplier < 0.0:
            raise ValueError("geometry_aware_ce.separator_multiplier must be non-negative.")
        if not self.separator_radius_multipliers:
            raise ValueError(
                "geometry_aware_ce.separator_radius_multipliers must not be empty."
            )
        if any(value <= 0.0 for value in self.separator_radius_multipliers):
            raise ValueError(
                "geometry_aware_ce.separator_radius_multipliers must be positive."
            )

    def __call__(self, target: torch.Tensor | np.ndarray) -> torch.Tensor:
        target_array = (
            target.detach().cpu().numpy()
            if isinstance(target, torch.Tensor)
            else np.asarray(target)
        )
        target_array = np.squeeze(target_array)
        if target_array.ndim != 2:
            raise ValueError(
                "Geometry-aware target weights require a 2D class-index mask, "
                f"got shape {tuple(target_array.shape)}."
            )
        return torch.from_numpy(self.build_numpy(target_array))

    def build_numpy(self, target: np.ndarray) -> np.ndarray:
        from scipy.ndimage import distance_transform_edt
        from skimage.morphology import closing, disk, skeletonize

        loci = target == LOCI_CLASS
        background = target == BACKGROUND_CLASS
        inoculum = target == INOCULUM_CLASS
        weights = np.ones(target.shape, dtype=np.float32)
        if not bool(loci.any()):
            return weights

        distance = distance_transform_edt(loci).astype(np.float32, copy=False)
        skeleton = skeletonize(loci)
        if not bool(skeleton.any()):
            return weights

        skeleton_radius = distance * skeleton
        _, nearest_indices = distance_transform_edt(
            ~skeleton,
            return_distances=True,
            return_indices=True,
        )
        nearest_radius = skeleton_radius[tuple(nearest_indices)]
        contour_depth = np.zeros(target.shape, dtype=np.float32)
        contour_depth[loci] = np.clip(
            distance[loci] / np.maximum(nearest_radius[loci], 1.0e-6),
            0.0,
            1.0,
        )
        weights[loci] += self.center_multiplier * np.power(
            contour_depth[loci], self.center_gamma
        )

        if self.separator_enabled and self.separator_multiplier > 0.0:
            positive_radii = skeleton_radius[skeleton]
            median_radius = float(np.median(positive_radii))
            radii = sorted(
                {
                    max(1, int(round(median_radius * multiplier)))
                    for multiplier in self.separator_radius_multipliers
                }
            )
            separator = np.zeros(target.shape, dtype=bool)
            for radius in radii:
                padded_loci = np.pad(
                    loci, radius, mode="constant", constant_values=False
                )
                padded_closed = closing(
                    padded_loci,
                    footprint=disk(radius, decomposition="crosses"),
                )
                closed = padded_closed[radius:-radius, radius:-radius]
                separator |= closed & ~loci
            separator &= background
            separator &= ~inoculum
            weights[separator] += self.separator_multiplier

        return weights


def build_geometry_weight_map_builder(
    loss_config: dict[str, Any],
) -> GeometryWeightMapBuilder | None:
    if str(loss_config.get("name", "")).strip().lower() != (
        "multiclass_geometry_ce_dice_loci_cldice"
    ):
        return None
    geometry_config = loss_config.get("geometry_aware_ce", {})
    if not isinstance(geometry_config, dict):
        raise ValueError("loss.geometry_aware_ce must be a mapping.")
    return GeometryWeightMapBuilder.from_config(geometry_config)
