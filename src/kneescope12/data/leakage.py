"""I flag potential grouping and duplication risks before cross-validation is designed."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LeakageAudit:
    """I separate verified duplicate content from weaker review candidates."""

    studies_with_multiple_series: dict[str, list[str | None]]
    exact_duplicate_volume_groups: dict[str, list[str]]
    metadata_similar_series_groups: dict[str, list[str]]
    patient_overlap_groups: dict[str, list[str]]

    def to_dict(self) -> dict[str, Any]:
        """I return a machine-readable leakage record without exposing image data."""

        return asdict(self)


def hash_volume(volume: np.ndarray) -> str:
    """I create a byte-level content hash for a reconstructed or cached volume."""

    contiguous = np.ascontiguousarray(volume)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def audit_leakage(
    manifest_rows: Iterable[Mapping[str, Any]],
    *,
    volume_hashes_by_series: Mapping[str, str] | None = None,
    patient_group_by_study: Mapping[str, str] | None = None,
) -> LeakageAudit:
    """I identify duplicate or grouping risks without proposing a data split.

    Matching geometry and description alone is only a review candidate, not proof
    of duplicated image content. Exact groups require caller-supplied volume hashes.
    """

    rows = list(manifest_rows)
    series_by_study: dict[str, list[str | None]] = defaultdict(list)
    metadata_groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        study_id = row.get("study_id")
        series_id = row.get("series_id")
        if isinstance(study_id, str):
            series_by_study[study_id].append(series_id if isinstance(series_id, str) else None)
        if isinstance(series_id, str):
            signature = _metadata_signature(row)
            metadata_groups[signature].append(series_id)
    exact_groups: dict[str, list[str]] = defaultdict(list)
    if volume_hashes_by_series is not None:
        for series_id, digest in volume_hashes_by_series.items():
            exact_groups[digest].append(series_id)
    patient_groups: dict[str, list[str]] = defaultdict(list)
    if patient_group_by_study is not None:
        for study_id, group_id in patient_group_by_study.items():
            patient_groups[group_id].append(study_id)
    return LeakageAudit(
        studies_with_multiple_series={
            study: sorted(series for series in series_ids if series is not None)
            for study, series_ids in sorted(series_by_study.items())
            if len(series_ids) > 1
        },
        exact_duplicate_volume_groups={
            digest: sorted(series_ids)
            for digest, series_ids in sorted(exact_groups.items())
            if len(series_ids) > 1
        },
        metadata_similar_series_groups={
            signature: sorted(series_ids)
            for signature, series_ids in sorted(metadata_groups.items())
            if len(series_ids) > 1
        },
        patient_overlap_groups={
            group: sorted(studies) for group, studies in sorted(patient_groups.items()) if len(studies) > 1
        },
    )


def _metadata_signature(row: Mapping[str, Any]) -> str:
    """I create a review-only signature from non-identifying manifest geometry fields."""

    fields = (
        row.get("series_description"),
        row.get("num_slices"),
        tuple(row.get("pixel_spacing_mm") or ()),
        row.get("slice_spacing_mm"),
        tuple(row.get("orientation") or ()),
    )
    return hashlib.sha256(repr(fields).encode("utf-8")).hexdigest()[:16]
