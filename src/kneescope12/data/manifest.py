"""I write minimal study-series manifests for later grouped model development."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(frozen=True)
class StudyManifestRow:
    """I retain only series-level fields needed to join cached MRI data later."""

    study_id: str | None
    series_id: str | None
    series_description: str | None
    laterality: str | None
    num_slices: int
    rows: int | None
    columns: int | None
    pixel_spacing_mm: tuple[float | None, float | None]
    slice_spacing_mm: float | None
    orientation: tuple[float, float, float, float, float, float] | None
    volume_path: str
    roi_path: str
    quality_flags: tuple[str, ...]
    preprocessing_version: str
    preprocessing_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        """I return an Arrow-compatible row while preserving null values."""

        result = asdict(self)
        result["pixel_spacing_mm"] = list(self.pixel_spacing_mm)
        result["orientation"] = list(self.orientation) if self.orientation is not None else None
        result["quality_flags"] = list(self.quality_flags)
        return result


def write_manifest(rows: Iterable[StudyManifestRow], path: str | Path) -> Path:
    """I write a Parquet manifest with no patient name, report text, or image pixels."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    materialized = [row.to_dict() for row in rows]
    table = pa.Table.from_pylist(materialized, schema=_manifest_schema())
    pq.write_table(table, destination, compression="zstd")
    return destination


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    """I read manifest rows for QC, leakage checks, and later grouped training joins."""

    return pq.read_table(path).to_pylist()


def _manifest_schema() -> pa.Schema:
    """I define a stable minimal schema for the study-series manifest."""

    return pa.schema(
        [
            pa.field("study_id", pa.string()),
            pa.field("series_id", pa.string()),
            pa.field("series_description", pa.string()),
            pa.field("laterality", pa.string()),
            pa.field("num_slices", pa.int32()),
            pa.field("rows", pa.int32()),
            pa.field("columns", pa.int32()),
            pa.field("pixel_spacing_mm", pa.list_(pa.float64())),
            pa.field("slice_spacing_mm", pa.float64()),
            pa.field("orientation", pa.list_(pa.float64())),
            pa.field("volume_path", pa.string()),
            pa.field("roi_path", pa.string()),
            pa.field("quality_flags", pa.list_(pa.string())),
            pa.field("preprocessing_version", pa.string()),
            pa.field("preprocessing_fingerprint", pa.string()),
        ]
    )
