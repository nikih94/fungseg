from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import torch
from torch.utils.data import WeightedRandomSampler

from src.patching import PatchRecord


def patch_distribution(records: list[PatchRecord]) -> dict[str, Any]:
    by_source = Counter(record.source_id for record in records)
    by_scale_label = Counter(record.scale_label for record in records)
    by_resolution_bucket = Counter(record.resolution_bucket for record in records)
    by_bucket_source = Counter(
        f"{record.resolution_bucket}::{record.source_id}" for record in records
    )
    return {
        "total_patches": len(records),
        "by_source": dict(sorted(by_source.items())),
        "by_scale_label": dict(sorted(by_scale_label.items())),
        "by_resolution_bucket": dict(sorted(by_resolution_bucket.items())),
        "by_resolution_bucket_source": dict(sorted(by_bucket_source.items())),
    }


def _resolve_samples_per_epoch(value: Any, records: list[PatchRecord]) -> int:
    if value is None or str(value).strip().lower() == "native_patch_count":
        native_count = sum(1 for record in records if record.scale_label in {"native", "normal"})
        return native_count or len(records)
    return int(value)


def build_balanced_resolution_source_sampler(
    records: list[PatchRecord],
    sampling_config: dict[str, Any],
    generator: torch.Generator | None = None,
) -> tuple[WeightedRandomSampler | None, dict[str, Any]]:
    strategy = str(sampling_config.get("strategy", "none")).strip().lower()
    if strategy in {"none", "natural", "shuffle"}:
        return None, {"enabled": False, "strategy": strategy}
    if strategy != "balanced_resolution_source":
        raise ValueError(f"Unsupported sampling strategy: {sampling_config.get('strategy')}")
    if not records:
        return None, {"enabled": False, "strategy": strategy, "reason": "empty_records"}

    bucket_source_counts = Counter(
        (record.resolution_bucket, record.source_id) for record in records
    )
    bucket_sources: dict[str, set[str]] = defaultdict(set)
    for bucket, source_id in bucket_source_counts:
        bucket_sources[bucket].add(source_id)

    weights = []
    for record in records:
        source_patch_count = bucket_source_counts[(record.resolution_bucket, record.source_id)]
        source_count = len(bucket_sources[record.resolution_bucket])
        weights.append(1.0 / max(source_patch_count * source_count, 1))

    samples_per_epoch = _resolve_samples_per_epoch(
        sampling_config.get("samples_per_epoch", "native_patch_count"),
        records,
    )
    replacement = bool(sampling_config.get("replacement", True))
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=samples_per_epoch,
        replacement=replacement,
        generator=generator,
    )

    total_weight = float(sum(weights))
    bucket_weight = defaultdict(float)
    source_weight = defaultdict(float)
    for record, weight in zip(records, weights):
        bucket_weight[record.resolution_bucket] += weight
        source_weight[f"{record.resolution_bucket}::{record.source_id}"] += weight

    effective_samples_by_bucket = {
        bucket: samples_per_epoch * (weight / total_weight)
        for bucket, weight in sorted(bucket_weight.items())
    }
    diagnostics = {
        "enabled": True,
        "strategy": strategy,
        "samples_per_epoch": samples_per_epoch,
        "replacement": replacement,
        "total_weight": total_weight,
        "weight_by_resolution_bucket": dict(sorted(bucket_weight.items())),
        "weight_by_resolution_bucket_source": dict(sorted(source_weight.items())),
        "effective_samples_per_bucket": effective_samples_by_bucket,
    }
    return sampler, diagnostics
