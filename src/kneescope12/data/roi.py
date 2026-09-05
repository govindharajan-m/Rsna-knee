"""I estimate a conservative image-derived candidate ROI with a full-volume fallback."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class RoiConfig:
    """I describe the deliberately simple initial foreground-based ROI estimator."""

    method: str = "foreground_quantile"
    foreground_quantile: float = 0.60
    margin_pixels: int = 12
    minimum_area_fraction: float = 0.02

    def __post_init__(self) -> None:
        """I validate estimator controls without implying anatomical correctness."""

        if self.method != "foreground_quantile":
            raise ValueError(f"Unsupported ROI method: {self.method}")
        if not 0 < self.foreground_quantile < 1:
            raise ValueError("foreground_quantile must fall strictly between zero and one")
        if self.margin_pixels < 0:
            raise ValueError("margin_pixels must not be negative")
        if not 0 < self.minimum_area_fraction <= 1:
            raise ValueError("minimum_area_fraction must fall in (0, 1]")


@dataclass(frozen=True)
class RoiResult:
    """I retain one candidate local ROI and the evidence or fallback behind it."""

    local_volume: np.ndarray
    bounding_box_zyx: tuple[int, int, int, int, int, int]
    source_shape: tuple[int, int, int]
    margin_pixels: int
    confidence: float
    quality_flag: str
    fallback_used: bool

    def metadata_record(self) -> dict[str, object]:
        """I return JSON-ready ROI metadata without duplicating voxel values."""

        return {
            "bounding_box_zyx": self.bounding_box_zyx,
            "source_shape": self.source_shape,
            "margin_pixels": self.margin_pixels,
            "confidence": self.confidence,
            "quality_flag": self.quality_flag,
            "fallback_used": self.fallback_used,
        }


def estimate_roi(volume: np.ndarray, config: RoiConfig) -> RoiResult:
    """I locate non-background in-plane support and return a candidate local volume.

    I use this only as localization groundwork. If the threshold produces too
    little, too much, or invalid foreground, I safely return the complete volume.
    """

    array = np.asarray(volume, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"I expect a three-dimensional volume, received {array.shape}")
    full_box = (0, array.shape[0], 0, array.shape[1], 0, array.shape[2])
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return _fallback(array, full_box, config, "roi_fallback_no_finite_voxels")
    threshold = float(np.quantile(finite, config.foreground_quantile))
    support = np.any(array > threshold, axis=0)
    coordinates = np.argwhere(support)
    if coordinates.size == 0:
        return _fallback(array, full_box, config, "roi_fallback_empty_foreground")
    y_start, x_start = coordinates.min(axis=0)
    y_end, x_end = coordinates.max(axis=0) + 1
    area_fraction = ((y_end - y_start) * (x_end - x_start)) / (array.shape[1] * array.shape[2])
    if area_fraction < config.minimum_area_fraction or area_fraction >= 0.98:
        return _fallback(array, full_box, config, "roi_fallback_unreliable_foreground")
    y_start = max(0, int(y_start) - config.margin_pixels)
    x_start = max(0, int(x_start) - config.margin_pixels)
    y_end = min(array.shape[1], int(y_end) + config.margin_pixels)
    x_end = min(array.shape[2], int(x_end) + config.margin_pixels)
    box = (0, array.shape[0], y_start, y_end, x_start, x_end)
    confidence = float(min(1.0, area_fraction / max(config.minimum_area_fraction, 1e-6)))
    return RoiResult(
        local_volume=array[:, y_start:y_end, x_start:x_end],
        bounding_box_zyx=box,
        source_shape=tuple(int(value) for value in array.shape),
        margin_pixels=config.margin_pixels,
        confidence=confidence,
        quality_flag="roi_candidate_image_derived",
        fallback_used=False,
    )


def roi_config_record(config: RoiConfig) -> dict[str, object]:
    """I return a stable configuration mapping for cache fingerprints."""

    return asdict(config)


def _fallback(
    volume: np.ndarray,
    full_box: tuple[int, int, int, int, int, int],
    config: RoiConfig,
    flag: str,
) -> RoiResult:
    """I return the complete volume when initial ROI evidence is insufficient."""

    return RoiResult(
        local_volume=volume.copy(),
        bounding_box_zyx=full_box,
        source_shape=tuple(int(value) for value in volume.shape),
        margin_pixels=config.margin_pixels,
        confidence=0.0,
        quality_flag=flag,
        fallback_used=True,
    )
