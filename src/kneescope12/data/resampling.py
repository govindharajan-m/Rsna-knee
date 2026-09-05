"""I resample MRI volumes only from explicit spacing and size configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import zoom


@dataclass(frozen=True)
class ResamplingConfig:
    """I specify optional target voxel spacing and in-plane image dimensions."""

    target_spacing_mm: tuple[float, float, float] | None = None
    target_size: tuple[int, int] | None = None
    interpolation_order: int = 1

    def __post_init__(self) -> None:
        """I validate that requested geometry is physically and numerically meaningful."""

        if self.target_spacing_mm is not None and any(value <= 0 for value in self.target_spacing_mm):
            raise ValueError("Target spacing values must be positive")
        if self.target_size is not None and any(value <= 0 for value in self.target_size):
            raise ValueError("Target image size values must be positive")
        if self.interpolation_order not in {0, 1, 2, 3, 4, 5}:
            raise ValueError("Interpolation order must be between zero and five")


@dataclass(frozen=True)
class ResamplingResult:
    """I retain a resampled volume and the relationship to its input geometry."""

    volume: np.ndarray
    input_shape: tuple[int, int, int]
    output_shape: tuple[int, int, int]
    input_spacing_mm: tuple[float | None, float | None, float | None]
    output_spacing_mm: tuple[float | None, float | None, float | None]
    scale_factors: tuple[float, float, float]
    applied: bool

    def config_record(self) -> dict[str, object]:
        """I return JSON-ready spatial-transform metadata."""

        return asdict(self)


def resample_volume(
    volume: np.ndarray,
    *,
    input_spacing_mm: tuple[float | None, float | None, float | None],
    config: ResamplingConfig,
) -> ResamplingResult:
    """I apply requested spacing and in-plane size transforms in one interpolation step."""

    array = np.asarray(volume, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"I expect a three-dimensional volume, received {array.shape}")
    scale = np.ones(3, dtype=float)
    output_spacing = list(input_spacing_mm)
    if config.target_spacing_mm is not None:
        if any(value is None for value in input_spacing_mm):
            raise ValueError("I need complete input spacing to resample to target spacing")
        source_spacing = np.asarray(input_spacing_mm, dtype=float)
        target_spacing = np.asarray(config.target_spacing_mm, dtype=float)
        scale = source_spacing / target_spacing
        output_spacing = list(config.target_spacing_mm)
    if config.target_size is not None:
        target_rows, target_columns = config.target_size
        scale[1] = target_rows / array.shape[1]
        scale[2] = target_columns / array.shape[2]
        if output_spacing[1] is not None:
            output_spacing[1] = float(input_spacing_mm[1]) / scale[1]
        if output_spacing[2] is not None:
            output_spacing[2] = float(input_spacing_mm[2]) / scale[2]
    applied = not np.allclose(scale, 1.0)
    output = zoom(array, zoom=tuple(scale), order=config.interpolation_order) if applied else array.copy()
    return ResamplingResult(
        volume=output,
        input_shape=tuple(int(value) for value in array.shape),
        output_shape=tuple(int(value) for value in output.shape),
        input_spacing_mm=input_spacing_mm,
        output_spacing_mm=tuple(output_spacing),
        scale_factors=tuple(float(value) for value in scale),
        applied=applied,
    )


def resampling_config_record(config: ResamplingConfig) -> dict[str, object]:
    """I return a stable configuration mapping for cache fingerprints."""

    return asdict(config)
