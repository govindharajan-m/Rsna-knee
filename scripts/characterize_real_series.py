"""I characterize a bounded, deterministic sample of real RSNA DICOM series."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from kneescope12.data.dicom import (  # noqa: E402, I001
    DicomMetadata,
    order_dicom_slices,
    read_dicom_metadata,
)


MAX_GROUPS = 20
MAX_SERIES_PER_GROUP = 3
MAX_SELECTED_SERIES = MAX_GROUPS * MAX_SERIES_PER_GROUP
CSV_FIELDS = {
    "study_id": "StudyInstanceUID",
    "series_id": "SeriesInstanceUID",
    "fluid_sensitive": "Fluid_Sensitive",
    "fat_suppression": "Fat_Suppression",
    "anatomical_plane": "Anatomical_Plane",
}
GEOMETRY_FIELDS = {
    "ImagePositionPatient": "image_position_patient",
    "ImageOrientationPatient": "image_orientation_patient",
    "PixelSpacing": "pixel_spacing",
    "SliceThickness": "slice_thickness",
    "Rows": "rows",
    "Columns": "columns",
}


@dataclass(frozen=True)
class SeriesGroup:
    """I retain manifest series sharing one study, plane, and acquisition family."""

    study_id: str
    anatomical_plane: str
    fluid_sensitive: str
    fat_suppression: str
    rows: tuple[dict[str, str], ...]

    @property
    def family(self) -> tuple[str, str, str]:
        """I return the CSV family labels used to interleave groups."""

        return self.anatomical_plane, self.fluid_sensitive, self.fat_suppression


def _parse_args() -> argparse.Namespace:
    """I parse the dataset location and optional report destination."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(os.environ["KNEESCOPE_DATA_ROOT"])
        if os.environ.get("KNEESCOPE_DATA_ROOT")
        else None,
        help="RSNA dataset root; defaults to KNEESCOPE_DATA_ROOT when set",
    )
    parser.add_argument("--series-csv", type=Path, help="Optional explicit train_series.csv path")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    return parser.parse_args()


def _find_series_csv(dataset_root: Path, explicit: Path | None) -> Path:
    """I locate train_series.csv without traversing the DICOM hierarchy."""

    candidates = [explicit] if explicit is not None else [dataset_root / "train_series.csv"]
    if explicit is None and dataset_root.name == "train_images":
        candidates.append(dataset_root.parent / "train_series.csv")
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"I could not find train_series.csv at: {searched}")


def _read_manifest(series_csv: Path) -> list[dict[str, str]]:
    """I read the five RSNA grouping columns from the small series manifest."""

    with series_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"I found no header in {series_csv}")
        normalized = {name.strip().casefold(): name for name in reader.fieldnames}
        missing = [
            column
            for column in CSV_FIELDS.values()
            if column.casefold() not in normalized
        ]
        if missing:
            raise ValueError(f"I could not find required RSNA train_series.csv columns: {missing}")
        rows = []
        for row in reader:
            values = {
                key: (row.get(normalized[column.casefold()]) or "").strip()
                for key, column in CSV_FIELDS.items()
            }
            if values["study_id"] and values["series_id"]:
                rows.append(values)
    return rows


def _group_manifest(rows: list[dict[str, str]]) -> list[SeriesGroup]:
    """I group manifest rows by study, plane, and fluid/fat-suppression family."""

    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row["study_id"],
            row["anatomical_plane"],
            row["fluid_sensitive"],
            row["fat_suppression"],
        )
        groups[key].append(row)
    return [
        SeriesGroup(*key, tuple(sorted(group, key=lambda row: row["series_id"])))
        for key, group in sorted(groups.items())
    ]


def _interleave_groups(groups: list[SeriesGroup]) -> list[SeriesGroup]:
    """I deterministically round-robin study groups across acquisition families."""

    families: dict[tuple[str, str, str], list[SeriesGroup]] = defaultdict(list)
    for group in groups:
        families[group.family].append(group)
    family_keys = sorted(families)
    result: list[SeriesGroup] = []
    index = 0
    while True:
        added = False
        for family in family_keys:
            family_groups = families[family]
            if index < len(family_groups):
                result.append(family_groups[index])
                added = True
        if not added:
            return result
        index += 1


def _select_groups(groups: list[SeriesGroup]) -> list[SeriesGroup]:
    """I prioritize multi-series groups while bounding selected series to 60."""

    multiple = _interleave_groups([group for group in groups if len(group.rows) > 1])
    single = _interleave_groups([group for group in groups if len(group.rows) == 1])
    selected: list[SeriesGroup] = []
    selected_count = 0
    for group in multiple + single:
        available = min(len(group.rows), MAX_SERIES_PER_GROUP)
        if selected_count + available > MAX_SELECTED_SERIES:
            available = MAX_SELECTED_SERIES - selected_count
        if available <= 0:
            break
        selected.append(group)
        selected_count += available
        if len(selected) == MAX_GROUPS or selected_count == MAX_SELECTED_SERIES:
            break
    return selected


def _distribution(values: list[int | float]) -> dict[str, int | float | None]:
    """I return sample count, minimum, median, and maximum for numeric values."""

    if not values:
        return {"count": 0, "minimum": None, "median": None, "maximum": None}
    return {
        "count": len(values),
        "minimum": min(values),
        "median": median(values),
        "maximum": max(values),
    }


def _summarize_metadata(metadata: list[DicomMetadata]) -> dict[str, object]:
    """I summarize metadata fields without materializing image pixels."""

    field_counts = {
        name: {
            "present": sum(getattr(item, attribute) is not None for item in metadata),
            "absent": sum(getattr(item, attribute) is None for item in metadata),
        }
        for name, attribute in GEOMETRY_FIELDS.items()
    }
    spacings = [item.pixel_spacing for item in metadata if item.pixel_spacing is not None]
    thicknesses = [
        item.slice_thickness for item in metadata if item.slice_thickness is not None
    ]
    rows = [item.rows for item in metadata if item.rows is not None]
    columns = [item.columns for item in metadata if item.columns is not None]
    dimensions = Counter(
        f"{item.rows}x{item.columns}"
        for item in metadata
        if item.rows is not None and item.columns is not None
    )
    pixel_spacing = {
        "row_mm": median([spacing[0] for spacing in spacings]) if spacings else None,
        "column_mm": median([spacing[1] for spacing in spacings]) if spacings else None,
    }
    return {
        "metadata_read_slices": len(metadata),
        "geometry_metadata_presence": field_counts,
        "pixel_spacing_mm": pixel_spacing,
        "slice_thickness_mm": median(thicknesses) if thicknesses else None,
        "image_rows": median(rows) if rows else None,
        "image_columns": median(columns) if columns else None,
        "image_dimensions": dict(sorted(dimensions.items())),
    }


def _series_directory(dataset_root: Path, study_id: str, series_id: str) -> Path:
    """I resolve only a sampled series under the canonical train_series hierarchy."""

    return dataset_root / "train_series" / study_id / series_id


def _characterize_series(
    dataset_root: Path, row: dict[str, str], group_size: int, selected_from_group: int
) -> dict[str, object]:
    """I characterize one selected series using metadata and shared ordering logic."""

    directory = _series_directory(dataset_root, row["study_id"], row["series_id"])
    paths = (
        tuple(sorted(path for path in directory.iterdir() if path.is_file()))
        if directory.is_dir()
        else ()
    )
    record: dict[str, object] = {
        "study_id": row["study_id"],
        "series_id": row["series_id"],
        "anatomical_plane": row["anatomical_plane"],
        "fluid_sensitive": row["fluid_sensitive"],
        "fat_suppression": row["fat_suppression"],
        "group_size": group_size,
        "selected_series_from_group": selected_from_group,
        "slice_count": len(paths),
        "geometry_ordering_succeeds": False,
        "geometry_ordering_error": None,
        "physical_adjacent_spacing_mm": [],
        "physical_extent_mm": None,
        "duplicate_physical_coordinates": None,
    }
    if not paths:
        record["geometry_ordering_error"] = (
            "Series directory is missing or contains no files"
        )
        record.update(_summarize_metadata([]))
        record["metadata_unreadable_slices"] = 0
        return record

    try:
        ordering = order_dicom_slices(paths)
        metadata = [item.metadata for item in ordering.slices]
        coordinates = [item.physical_coordinate for item in ordering.slices]
        spacings = [
            abs(right - left)
            for left, right in zip(coordinates, coordinates[1:], strict=False)
        ]
        record["geometry_ordering_succeeds"] = True
        record["physical_adjacent_spacing_mm"] = spacings
        record["physical_extent_mm"] = max(coordinates) - min(coordinates)
        record["duplicate_physical_coordinates"] = (
            ordering.has_duplicate_physical_coordinates
        )
        record["metadata_unreadable_slices"] = 0
    except (OSError, ValueError) as error:
        metadata = []
        unreadable = 0
        for path in paths:
            result = read_dicom_metadata(path)
            if result.metadata is None:
                unreadable += 1
            else:
                metadata.append(result.metadata)
        record["geometry_ordering_error"] = f"{type(error).__name__}: {error}"
        record["metadata_unreadable_slices"] = unreadable
    record.update(_summarize_metadata(metadata))
    return record


def characterize(dataset_root: Path, series_csv: Path | None = None) -> dict[str, object]:
    """I characterize a reproducible, bounded set of manifest groups and series."""

    csv_path = _find_series_csv(dataset_root, series_csv)
    rows = _read_manifest(csv_path)
    groups = _group_manifest(rows)
    selected_groups = _select_groups(groups)
    family_counts: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    for group in groups:
        counts = family_counts[group.family]
        counts["groups"] += 1
        counts["series"] += len(group.rows)
        if len(group.rows) > 1:
            counts["multiple_series_groups"] += 1

    selected_rows: list[tuple[SeriesGroup, dict[str, str]]] = []
    for group in selected_groups:
        for row in group.rows[:MAX_SERIES_PER_GROUP]:
            selected_rows.append((group, row))
    series_reports = [
        _characterize_series(
            dataset_root,
            row,
            len(group.rows),
            min(len(group.rows), MAX_SERIES_PER_GROUP),
        )
        for group, row in selected_rows
    ]
    numeric_metrics: dict[str, list[int | float]] = {
        "slice_count": [],
        "pixel_spacing_row_mm": [],
        "pixel_spacing_column_mm": [],
        "slice_thickness_mm": [],
        "image_rows": [],
        "image_columns": [],
        "physical_adjacent_spacing_mm": [],
        "physical_extent_mm": [],
    }
    geometry_presence = {
        name: {"present": 0, "absent": 0}
        for name in GEOMETRY_FIELDS
    }
    dimensions: Counter[str] = Counter()
    for series in series_reports:
        numeric_metrics["slice_count"].append(series["slice_count"])
        for field in GEOMETRY_FIELDS:
            field_counts = series["geometry_metadata_presence"][field]
            geometry_presence[field]["present"] += field_counts["present"]
            geometry_presence[field]["absent"] += field_counts["absent"]
        dimensions.update(series["image_dimensions"])
        for metric, key in (
            ("pixel_spacing_row_mm", "row_mm"),
            ("pixel_spacing_column_mm", "column_mm"),
        ):
            value = series["pixel_spacing_mm"][key]
            if value is not None:
                numeric_metrics[metric].append(value)
        for metric, key in (
            ("slice_thickness_mm", "slice_thickness_mm"),
            ("image_rows", "image_rows"),
            ("image_columns", "image_columns"),
            ("physical_extent_mm", "physical_extent_mm"),
        ):
            value = series[key]
            if value is not None:
                numeric_metrics[metric].append(value)
        numeric_metrics["physical_adjacent_spacing_mm"].extend(
            series["physical_adjacent_spacing_mm"]
        )
    family_reports = [
        {
            "anatomical_plane": family[0],
            "fluid_sensitive": family[1],
            "fat_suppression": family[2],
            "groups": counts["groups"],
            "series": counts["series"],
            "multiple_series_groups": counts["multiple_series_groups"],
        }
        for family, counts in sorted(family_counts.items())
    ]
    selected_group_reports = [
        {
            "study_id": group.study_id,
            "anatomical_plane": group.anatomical_plane,
            "fluid_sensitive": group.fluid_sensitive,
            "fat_suppression": group.fat_suppression,
            "manifest_series_count": len(group.rows),
            "selected_series_count": min(len(group.rows), MAX_SERIES_PER_GROUP),
            "truncated": len(group.rows) > MAX_SERIES_PER_GROUP,
        }
        for group in selected_groups
    ]
    return {
        "dataset_root": str(dataset_root),
        "series_csv": str(csv_path),
        "sampling": {
            "strategy": (
                "Deterministic family-interleaved selection; multi-series groups first; "
                "series IDs sorted within each group"
            ),
            "manifest_series_count": len(rows),
            "manifest_group_count": len(groups),
            "multi_series_group_count": sum(len(group.rows) > 1 for group in groups),
            "max_selected_groups": MAX_GROUPS,
            "max_series_per_group": MAX_SERIES_PER_GROUP,
            "max_selected_series": MAX_SELECTED_SERIES,
            "selected_group_count": len(selected_groups),
            "selected_series_count": len(series_reports),
        },
        "group_counts_by_acquisition_family": family_reports,
        "selected_groups": selected_group_reports,
        "series_characteristics": series_reports,
        "geometry_ordering_success_count": sum(
            series["geometry_ordering_succeeds"] for series in series_reports
        ),
        "geometry_ordering_failure_count": sum(
            not series["geometry_ordering_succeeds"] for series in series_reports
        ),
        "geometry_metadata_presence": geometry_presence,
        "image_dimensions_counts": dict(sorted(dimensions.items())),
        "metric_distributions": {
            metric: _distribution(values) for metric, values in numeric_metrics.items()
        },
    }


def main() -> int:
    """I run the diagnostic and write its deterministic JSON report."""

    arguments = _parse_args()
    if arguments.dataset_root is None:
        print("SERIES CHARACTERIZATION SKIPPED: set --dataset-root or KNEESCOPE_DATA_ROOT")
        return 0
    dataset_root = arguments.dataset_root.expanduser().resolve()
    try:
        report = characterize(dataset_root, arguments.series_csv)
    except (FileNotFoundError, ValueError) as error:
        print(f"SERIES CHARACTERIZATION FAILED: {error}")
        return 1
    if arguments.output is not None:
        output = arguments.output.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["geometry_ordering_failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
