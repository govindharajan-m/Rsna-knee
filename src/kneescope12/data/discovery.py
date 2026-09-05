"""I discover DICOM series without assuming a competition directory layout."""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .audit import DatasetAuditConfig
from .dicom import DicomMetadata, read_dicom_metadata


@dataclass(frozen=True)
class DiscoveredSeries:
    """I group readable DICOM metadata by observed study and series identifiers."""

    study_id: str | None
    series_id: str | None
    slices: tuple[DicomMetadata, ...]

    @property
    def modality(self) -> str | None:
        """I return the common modality only when the series metadata agrees."""

        values = {item.modality for item in self.slices if item.modality is not None}
        return next(iter(values)) if len(values) == 1 else None


def discover_series(
    data_root: str | Path,
    *,
    config: DatasetAuditConfig | None = None,
) -> tuple[DiscoveredSeries, ...]:
    """I read metadata lazily and group DICOM files for sequential reconstruction.

    I retain metadata rather than image pixels. Cache preparation then reconstructs
    one series at a time, so raw image arrays are never accumulated for a dataset.
    """

    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root is not a directory: {root}")
    scan_config = config or DatasetAuditConfig()
    grouped: dict[tuple[str, str], list[DicomMetadata]] = defaultdict(list)
    for path in _iter_files(root):
        if not scan_config.is_candidate(path):
            continue
        result = read_dicom_metadata(path)
        if result.metadata is None:
            continue
        metadata = result.metadata
        study_key = metadata.study_instance_uid or "<missing-study-uid>"
        series_key = metadata.series_instance_uid or f"<missing-series-uid:{path.relative_to(root)}>"
        grouped[(study_key, series_key)].append(metadata)
    return tuple(
        DiscoveredSeries(
            study_id=None if study_key == "<missing-study-uid>" else study_key,
            series_id=None if series_key.startswith("<missing-series-uid:") else series_key,
            slices=tuple(slices),
        )
        for (study_key, series_key), slices in sorted(grouped.items())
    )


def _iter_files(root: Path) -> Iterable[Path]:
    """I yield only regular files and hold one directory listing at a time."""

    for directory, directories, filenames in os.walk(root):
        directories.sort()
        filenames.sort()
        for filename in filenames:
            path = Path(directory) / filename
            if path.is_file():
                yield path
