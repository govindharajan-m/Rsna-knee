"""I build reproducible MRI preprocessing caches and manifests one series at a time."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from kneescope12.utils.logging import get_logger

from .audit import DatasetAuditConfig
from .cache import PreprocessingCache
from .discovery import discover_series
from .intensity import IntensityStatistics, measure_intensity, summarize_intensity
from .leakage import audit_leakage, hash_volume
from .manifest import StudyManifestRow, write_manifest
from .preprocessing import PreprocessingConfig, preprocess_volume
from .visual_qc import save_preprocessing_montage
from .volume import VolumeReconstruction, reconstruct_series

logger = get_logger("prepare")


@dataclass(frozen=True)
class PreparationResult:
    """I report generated artifact locations and processing counts without model metrics."""

    manifest_path: Path
    leakage_path: Path
    intensity_path: Path
    processed_series_count: int
    skipped_series_count: int
    cache_hit_count: int
    qc_paths: tuple[Path, ...]


def prepare_dataset(
    data_root: str | Path,
    *,
    cache_root: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    preprocessing: PreprocessingConfig,
    audit_config: DatasetAuditConfig | None = None,
    qc_samples: int = 0,
    max_series: int | None = None,
) -> PreparationResult:
    """I create a cache and manifest for observed MR series without selecting one per study."""

    if qc_samples < 0:
        raise ValueError("qc_samples must not be negative")
    if max_series is not None and max_series < 1:
        raise ValueError("max_series must be positive when supplied")
    artifacts = Path(output_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    cache = PreprocessingCache(cache_root, preprocessing)
    _write_preprocessing_record(cache.namespace, preprocessing)
    manifest_rows: list[StudyManifestRow] = []
    intensity_records: dict[str, dict[str, Any]] = {}
    volume_hashes: dict[str, str] = {}
    qc_paths: list[Path] = []
    processed = skipped = cache_hits = 0
    for series in discover_series(data_root, config=audit_config):
        if max_series is not None and processed + skipped >= max_series:
            break
        if series.modality != "MR":
            skipped += 1
            logger.info("I skipped non-MR or mixed-modality series %s", series.series_id)
            continue
        reconstruction = reconstruct_series(series.slices)
        if not reconstruction.is_usable or reconstruction.geometry is None:
            skipped += 1
            logger.warning("I skipped unusable MR series %s", series.series_id)
            continue
        geometry = reconstruction.geometry
        signature = cache.source_signature(reconstruction.source_paths)
        cached = cache.load(
            study_id=geometry.study_instance_uid,
            series_id=geometry.series_instance_uid,
            source_signature=signature,
        )
        if cached is None:
            processed_volume = preprocess_volume(reconstruction, preprocessing)
            row = _manifest_row(reconstruction, processed_volume, cache.cache_path(
                study_id=geometry.study_instance_uid, series_id=geometry.series_instance_uid
            ))
            cached = cache.store(
                study_id=geometry.study_instance_uid,
                series_id=geometry.series_instance_uid,
                source_signature=signature,
                processed=processed_volume,
                extra_metadata={"manifest_row": row.to_dict(), "raw_geometry": asdict(geometry)},
            )
            if len(qc_paths) < qc_samples:
                qc_path = artifacts / "qc" / f"{cached.cache_path.stem}.png"
                qc_paths.append(save_preprocessing_montage(reconstruction, processed_volume, qc_path))
        else:
            cache_hits += 1
            row = _manifest_row_from_dict(cached.metadata["manifest_row"])
        manifest_rows.append(row)
        processed += 1
        if geometry.series_instance_uid is not None:
            volume_hashes[geometry.series_instance_uid] = hash_volume(cached.global_volume)
            intensity_records[geometry.series_instance_uid] = measure_intensity(cached.global_volume).to_dict()

    manifest_destination = write_manifest(manifest_rows, manifest_path)
    leakage = audit_leakage(
        (row.to_dict() for row in manifest_rows), volume_hashes_by_series=volume_hashes
    )
    leakage_path = artifacts / "data_leakage.json"
    leakage_path.write_text(json.dumps(leakage.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    intensity_path = artifacts / "intensity_audit.json"
    intensity_summary = summarize_intensity(_statistics_from_dict(item) for item in intensity_records.values())
    intensity_path.write_text(
        json.dumps(
            {"summary": intensity_summary, "series": intensity_records}, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    return PreparationResult(
        manifest_path=manifest_destination,
        leakage_path=leakage_path,
        intensity_path=intensity_path,
        processed_series_count=processed,
        skipped_series_count=skipped,
        cache_hit_count=cache_hits,
        qc_paths=tuple(qc_paths),
    )


def _manifest_row(
    reconstruction: VolumeReconstruction,
    processed: Any,
    cache_path: Path,
) -> StudyManifestRow:
    """I derive one minimal manifest row after a successful reconstruction and transform."""

    if reconstruction.geometry is None or reconstruction.raw_volume is None:
        raise ValueError("I need a usable reconstruction to create a manifest row")
    geometry = reconstruction.geometry
    flags = tuple(sorted(set((*reconstruction.quality_flags, processed.roi.quality_flag))))
    return StudyManifestRow(
        study_id=geometry.study_instance_uid,
        series_id=geometry.series_instance_uid,
        series_description=geometry.series_description,
        laterality=geometry.laterality,
        num_slices=int(reconstruction.raw_volume.shape[0]),
        rows=geometry.original_dimensions[0] if geometry.original_dimensions is not None else None,
        columns=geometry.original_dimensions[1] if geometry.original_dimensions is not None else None,
        pixel_spacing_mm=(geometry.voxel_spacing_mm[1], geometry.voxel_spacing_mm[2]),
        slice_spacing_mm=geometry.voxel_spacing_mm[0],
        orientation=geometry.orientation,
        volume_path=str(cache_path),
        roi_path=f"{cache_path}#local_volume",
        quality_flags=flags,
        preprocessing_version=processed.preprocessing_version,
        preprocessing_fingerprint=processed.preprocessing_fingerprint,
    )


def _manifest_row_from_dict(values: Mapping[str, Any]) -> StudyManifestRow:
    """I reconstruct a typed manifest row from cache provenance after a cache hit."""

    return StudyManifestRow(
        study_id=values.get("study_id"),
        series_id=values.get("series_id"),
        series_description=values.get("series_description"),
        laterality=values.get("laterality"),
        num_slices=int(values["num_slices"]),
        rows=values.get("rows"),
        columns=values.get("columns"),
        pixel_spacing_mm=tuple(values["pixel_spacing_mm"]),
        slice_spacing_mm=values.get("slice_spacing_mm"),
        orientation=tuple(values["orientation"]) if values.get("orientation") is not None else None,
        volume_path=values["volume_path"],
        roi_path=values["roi_path"],
        quality_flags=tuple(values["quality_flags"]),
        preprocessing_version=values["preprocessing_version"],
        preprocessing_fingerprint=values["preprocessing_fingerprint"],
    )


def _write_preprocessing_record(namespace: Path, config: PreprocessingConfig) -> None:
    """I write the active preprocessing configuration beside the cache namespace."""

    namespace.mkdir(parents=True, exist_ok=True)
    (namespace / "preprocessing_config.json").write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _statistics_from_dict(values: Mapping[str, Any]) -> IntensityStatistics:
    """I reconstruct typed intensity statistics from the cache-audit record."""

    return IntensityStatistics(**dict(values))
