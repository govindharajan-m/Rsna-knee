"""I offer configurable MRI intensity transformations without choosing a final method."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from .intensity import IntensityStatistics, measure_intensity

NormalizationMethod = Literal["none", "percentile_minmax", "minmax", "zscore", "robust"]


@dataclass(frozen=True)
class NormalizationConfig:
    """I describe an explicit intensity transform for one preprocessing experiment."""

    method: NormalizationMethod = "percentile_minmax"
    lower_percentile: float = 0.5
    upper_percentile: float = 99.5
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        """I reject invalid percentile and denominator settings early."""

        if self.method not in {"none", "percentile_minmax", "minmax", "zscore", "robust"}:
            raise ValueError(f"Unsupported normalization method: {self.method}")
        if not 0 <= self.lower_percentile < self.upper_percentile <= 100:
            raise ValueError("Normalization percentiles must satisfy 0 <= lower < upper <= 100")
        if self.epsilon <= 0:
            raise ValueError("Normalization epsilon must be positive")


@dataclass(frozen=True)
class NormalizationResult:
    """I retain normalized voxels and the parameters that produced them."""

    volume: np.ndarray
    input_statistics: IntensityStatistics
    parameters: dict[str, float | str]

    def config_record(self) -> dict[str, object]:
        """I return JSON-ready transform metadata for cache provenance."""

        return {"input_statistics": self.input_statistics.to_dict(), "parameters": self.parameters}


def normalize_volume(volume: np.ndarray, config: NormalizationConfig) -> NormalizationResult:
    """I transform finite MRI intensities and preserve non-finite locations as zero.

    I support several common experimental transforms. I do not imply that any
    method is globally correct for MRI data.
    """

    array = np.asarray(volume, dtype=np.float32)
    statistics = measure_intensity(array)
    finite_mask = np.isfinite(array)
    output = np.zeros_like(array, dtype=np.float32)
    finite = array[finite_mask]
    parameters: dict[str, float | str] = {"method": config.method}
    if config.method == "none":
        output[finite_mask] = finite
    elif config.method == "minmax":
        lower, upper = float(np.min(finite)), float(np.max(finite))
        output[finite_mask] = (finite - lower) / max(upper - lower, config.epsilon)
        parameters.update({"lower": lower, "upper": upper})
    elif config.method == "percentile_minmax":
        lower, upper = np.percentile(finite, [config.lower_percentile, config.upper_percentile])
        clipped = np.clip(finite, lower, upper)
        output[finite_mask] = (clipped - lower) / max(float(upper - lower), config.epsilon)
        parameters.update(
            {
                "lower": float(lower),
                "upper": float(upper),
                "lower_percentile": config.lower_percentile,
                "upper_percentile": config.upper_percentile,
            }
        )
    elif config.method == "zscore":
        mean, standard_deviation = float(np.mean(finite)), float(np.std(finite))
        output[finite_mask] = (finite - mean) / max(standard_deviation, config.epsilon)
        parameters.update({"mean": mean, "standard_deviation": standard_deviation})
    else:
        median = float(np.median(finite))
        q25, q75 = np.percentile(finite, [25, 75])
        interquartile_range = float(q75 - q25)
        output[finite_mask] = (finite - median) / max(interquartile_range, config.epsilon)
        parameters.update({"median": median, "interquartile_range": interquartile_range})
    return NormalizationResult(volume=output, input_statistics=statistics, parameters=parameters)


def normalization_config_record(config: NormalizationConfig) -> dict[str, object]:
    """I return a stable configuration mapping for cache fingerprints."""

    return asdict(config)
