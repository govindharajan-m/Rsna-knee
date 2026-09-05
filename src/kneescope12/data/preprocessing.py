"""I compose explicit intensity, spatial, and ROI transforms for one raw volume."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

from .normalization import NormalizationConfig, NormalizationResult, normalize_volume
from .resampling import ResamplingConfig, ResamplingResult, resample_volume
from .roi import RoiConfig, RoiResult, estimate_roi
from .volume import VolumeReconstruction


@dataclass(frozen=True)
class PreprocessingConfig:
    """I version the transform choices that define a cache-compatible representation."""

    version: str
    normalization: NormalizationConfig
    resampling: ResamplingConfig
    roi: RoiConfig

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "PreprocessingConfig":
        """I construct preprocessing configuration from a YAML mapping."""

        if not isinstance(values, Mapping):
            raise ValueError("preprocessing configuration must be a mapping")
        version = values.get("version")
        if not isinstance(version, str) or not version.strip():
            raise ValueError("preprocessing.version must be a non-empty string")
        normalization_values = values.get("normalization", {})
        resampling_values = values.get("resampling", {})
        roi_values = values.get("roi", {})
        if not all(isinstance(item, Mapping) for item in (normalization_values, resampling_values, roi_values)):
            raise ValueError("preprocessing transform sections must be mappings")
        return cls(
            version=version,
            normalization=NormalizationConfig(**dict(normalization_values)),
            resampling=ResamplingConfig(**_coerce_resampling_values(resampling_values)),
            roi=RoiConfig(**dict(roi_values)),
        )

    def fingerprint(self) -> str:
        """I derive a stable hash that invalidates cache entries after configuration changes."""

        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, object]:
        """I return a JSON-ready preprocessing configuration record."""

        return {
            "version": self.version,
            "normalization": asdict(self.normalization),
            "resampling": asdict(self.resampling),
            "roi": asdict(self.roi),
        }


@dataclass(frozen=True)
class ProcessedVolume:
    """I distinguish processed global/local arrays from their raw reconstruction."""

    global_volume: np.ndarray
    local_volume: np.ndarray
    normalization: NormalizationResult
    resampling: ResamplingResult
    roi: RoiResult
    preprocessing_version: str
    preprocessing_fingerprint: str

    def metadata_record(self) -> dict[str, object]:
        """I provide cacheable metadata without serializing arrays into JSON."""

        return {
            "preprocessing_version": self.preprocessing_version,
            "preprocessing_fingerprint": self.preprocessing_fingerprint,
            "normalization": self.normalization.config_record(),
            "resampling": {
                "input_shape": self.resampling.input_shape,
                "output_shape": self.resampling.output_shape,
                "input_spacing_mm": self.resampling.input_spacing_mm,
                "output_spacing_mm": self.resampling.output_spacing_mm,
                "scale_factors": self.resampling.scale_factors,
                "applied": self.resampling.applied,
            },
            "roi": self.roi.metadata_record(),
        }


def preprocess_volume(
    reconstruction: VolumeReconstruction,
    config: PreprocessingConfig,
) -> ProcessedVolume:
    """I apply configured preprocessing while retaining raw geometry in the reconstruction."""

    if not reconstruction.is_usable or reconstruction.raw_volume is None or reconstruction.geometry is None:
        raise ValueError("I cannot preprocess an unusable reconstruction")
    normalized = normalize_volume(reconstruction.raw_volume, config.normalization)
    resampled = resample_volume(
        normalized.volume,
        input_spacing_mm=reconstruction.geometry.voxel_spacing_mm,
        config=config.resampling,
    )
    roi = estimate_roi(resampled.volume, config.roi)
    return ProcessedVolume(
        global_volume=resampled.volume,
        local_volume=roi.local_volume,
        normalization=normalized,
        resampling=resampled,
        roi=roi,
        preprocessing_version=config.version,
        preprocessing_fingerprint=config.fingerprint(),
    )


def _coerce_resampling_values(values: Mapping[str, Any]) -> dict[str, Any]:
    """I convert YAML lists into the immutable tuple fields used by resampling."""

    converted = dict(values)
    for key in ("target_spacing_mm", "target_size"):
        if converted.get(key) is not None:
            converted[key] = tuple(converted[key])
    return converted
