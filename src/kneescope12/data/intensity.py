"""I measure MRI intensity distributions before selecting a normalization strategy."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class IntensityStatistics:
    """I retain robust and conventional statistics for one finite-valued volume."""

    minimum: float
    maximum: float
    percentile_01: float
    percentile_05: float
    percentile_25: float
    median: float
    percentile_75: float
    percentile_95: float
    percentile_99: float
    interquartile_range: float
    robust_range: float
    finite_voxel_count: int

    def to_dict(self) -> dict[str, Any]:
        """I return a JSON-ready intensity record."""

        return asdict(self)


def measure_intensity(volume: np.ndarray) -> IntensityStatistics:
    """I calculate per-volume intensity statistics from finite voxels only."""

    finite = np.asarray(volume, dtype=np.float32)[np.isfinite(volume)]
    if finite.size == 0:
        raise ValueError("I cannot measure intensity because the volume has no finite voxels")
    percentiles = np.percentile(finite, [1, 5, 25, 50, 75, 95, 99])
    return IntensityStatistics(
        minimum=float(np.min(finite)),
        maximum=float(np.max(finite)),
        percentile_01=float(percentiles[0]),
        percentile_05=float(percentiles[1]),
        percentile_25=float(percentiles[2]),
        median=float(percentiles[3]),
        percentile_75=float(percentiles[4]),
        percentile_95=float(percentiles[5]),
        percentile_99=float(percentiles[6]),
        interquartile_range=float(percentiles[4] - percentiles[2]),
        robust_range=float(percentiles[5] - percentiles[1]),
        finite_voxel_count=int(finite.size),
    )


def summarize_intensity(records: Iterable[IntensityStatistics]) -> dict[str, float | int | None]:
    """I summarize between-volume variation without treating it as a normalization choice."""

    materialized = list(records)
    if not materialized:
        return {"volume_count": 0, "median_of_medians": None, "median_robust_range": None}
    medians = np.asarray([record.median for record in materialized], dtype=float)
    ranges = np.asarray([record.robust_range for record in materialized], dtype=float)
    return {
        "volume_count": len(materialized),
        "median_of_medians": float(np.median(medians)),
        "median_robust_range": float(np.median(ranges)),
        "between_volume_median_iqr": float(np.percentile(medians, 75) - np.percentile(medians, 25)),
    }
