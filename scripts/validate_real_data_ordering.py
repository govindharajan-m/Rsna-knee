"""I validate physical DICOM ordering on a deterministic real-data sample."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from kneescope12.data.dicom import DicomGeometryError, order_dicom_slices  # noqa: E402, I001


SAMPLE_SIZE = 20
CANDIDATE_POOL_SIZE = 60
CSV_FIELDS = {
    "study_id": "StudyInstanceUID",
    "series_id": "SeriesInstanceUID",
}
OPTIONAL_CSV_FIELDS = {
    "anatomical_plane": "anatomical_plane",
    "fluid_sensitive": "fluid_sensitive",
}


@dataclass(frozen=True)
class SeriesCandidate:
    """I retain one CSV-listed series and its bounded filesystem inspection."""

    study_id: str
    series_id: str
    anatomical_plane: str
    fluid_sensitive: str
    paths: tuple[Path, ...]

    @property
    def size(self) -> int:
        """I return the number of input DICOM files in this series directory."""

        return len(self.paths)


def _parse_args() -> argparse.Namespace:
    """I parse the explicit dataset location and optional report destination."""

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
    """I locate train_series.csv without recursively scanning the dataset."""

    candidates = [explicit] if explicit is not None else [dataset_root / "train_series.csv"]
    if explicit is None and dataset_root.name == "train_images":
        candidates.append(dataset_root.parent / "train_series.csv")
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"I could not find train_series.csv at: {searched}")


def _read_series_rows(series_csv: Path) -> list[dict[str, str]]:
    """I read the small series manifest and retain only required columns."""

    with series_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"I found no header in {series_csv}")
        normalized = {name.strip().lower(): name for name in reader.fieldnames}
        missing = [column for column in CSV_FIELDS.values() if column.lower() not in normalized]
        if missing:
            raise ValueError(f"I could not find required train_series.csv columns: {missing}")
        rows = []
        for row in reader:
            values = {
                key: (row.get(normalized[column.lower()]) or "").strip()
                for key, column in CSV_FIELDS.items()
            }
            values.update(
                {
                    key: (row.get(normalized.get(column.lower(), "")) or "").strip()
                    for key, column in OPTIONAL_CSV_FIELDS.items()
                }
            )
            if values["study_id"] and values["series_id"]:
                rows.append(values)
    return rows


def _series_directory(dataset_root: Path, study_id: str, series_id: str) -> Path:
    """I resolve one competition series under the canonical train_series root."""

    return dataset_root / "train_series" / study_id / series_id


def _csv_candidate_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """I create a deterministic, category-interleaved bounded candidate pool."""

    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["anatomical_plane"], row["fluid_sensitive"])].append(row)
    for group in groups.values():
        group.sort(key=lambda row: (row["study_id"], row["series_id"]))
    pool: list[dict[str, str]] = []
    keys = sorted(groups)
    index = 0
    while len(pool) < min(CANDIDATE_POOL_SIZE, len(rows)):
        added = False
        for key in keys:
            group = groups[key]
            if index < len(group):
                pool.append(group[index])
                added = True
                if len(pool) == CANDIDATE_POOL_SIZE:
                    break
        if not added:
            break
        index += 1
    return pool


def _candidate(row: dict[str, str], dataset_root: Path) -> SeriesCandidate | None:
    """I inspect one selected candidate directory without opening DICOM files."""

    directory = _series_directory(dataset_root, row["study_id"], row["series_id"])
    if not directory.is_dir():
        return None
    paths = tuple(sorted(path for path in directory.iterdir() if path.is_file()))
    if not paths:
        return None
    return SeriesCandidate(
        study_id=row["study_id"],
        series_id=row["series_id"],
        anatomical_plane=row["anatomical_plane"],
        fluid_sensitive=row["fluid_sensitive"],
        paths=paths,
    )


def _select_sample(candidates: list[SeriesCandidate]) -> list[SeriesCandidate]:
    """I interleave plane/fluid groups while spreading observed series sizes."""

    groups: dict[tuple[str, str], list[SeriesCandidate]] = defaultdict(list)
    for candidate in candidates:
        groups[(candidate.anatomical_plane, candidate.fluid_sensitive)].append(candidate)
    for group in groups.values():
        group.sort(key=lambda item: (item.size, item.study_id, item.series_id))
    selected: list[SeriesCandidate] = []
    keys = sorted(groups)
    index = 0
    while len(selected) < min(SAMPLE_SIZE, len(candidates)):
        added = False
        for key in keys:
            group = groups[key]
            if index < len(group):
                selected.append(group[index])
                added = True
                if len(selected) == SAMPLE_SIZE:
                    break
        if not added:
            break
        index += 1
    return selected


def _percentiles(values: list[int | float]) -> dict[str, int | float | None]:
    """I report minimum, median, and maximum without imposing a target value."""

    if not values:
        return {"minimum": None, "median": None, "maximum": None}
    return {"minimum": min(values), "median": median(values), "maximum": max(values)}


def validate(dataset_root: Path, series_csv: Path | None = None) -> dict[str, object]:
    """I validate one deterministic sample and return a JSON-compatible report."""

    csv_path = _find_series_csv(dataset_root, series_csv)
    rows = _read_series_rows(csv_path)
    candidates = [
        candidate
        for row in _csv_candidate_rows(rows)
        if (candidate := _candidate(row, dataset_root)) is not None
    ]
    sample = _select_sample(candidates)
    report: dict[str, object] = {
        "dataset_root": str(dataset_root),
        "series_csv": str(csv_path),
        "sample_size_requested": SAMPLE_SIZE,
        "sample_size_selected": len(sample),
        "series_successfully_validated": 0,
        "geometry_failure_count": 0,
        "geometry_failures": [],
        "duplicate_coordinate_series": 0,
        "slice_count": _percentiles([]),
        "adjacent_physical_spacing": _percentiles([]),
        "physical_extent": _percentiles([]),
        "selected_series": [],
    }
    slice_counts: list[int] = []
    spacings: list[float] = []
    extents: list[float] = []
    failures: list[dict[str, str]] = []
    selected_details: list[dict[str, object]] = []
    for candidate in sample:
        try:
            ordering = order_dicom_slices(candidate.paths)
            coordinates = [item.physical_coordinate for item in ordering.slices]
            if coordinates != sorted(coordinates):
                raise AssertionError("returned physical coordinates are not monotonically ordered")
            if len(ordering.slices) != len(candidate.paths):
                raise AssertionError("returned slice count differs from input DICOM file count")
            if ordering.has_duplicate_physical_coordinates:
                duplicate_count = sum(
                    item.duplicate_physical_coordinate for item in ordering.slices
                )
                if duplicate_count < 2:
                    raise AssertionError("duplicate coordinates were not retained and flagged")
            adjacent = [
                abs(right - left)
                for left, right in zip(coordinates, coordinates[1:], strict=False)
            ]
            extent = max(coordinates) - min(coordinates) if coordinates else 0.0
            slice_counts.append(len(candidate.paths))
            spacings.extend(adjacent)
            extents.append(extent)
            selected_details.append(
                {
                    "study_id": candidate.study_id,
                    "series_id": candidate.series_id,
                    "anatomical_plane": candidate.anatomical_plane,
                    "fluid_sensitive": candidate.fluid_sensitive,
                    "slice_count": len(candidate.paths),
                    "duplicate_physical_coordinates": ordering.has_duplicate_physical_coordinates,
                }
            )
        except (DicomGeometryError, AssertionError, OSError, ValueError) as error:
            failures.append(
                {
                    "study_id": candidate.study_id,
                    "series_id": candidate.series_id,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    report["series_successfully_validated"] = len(selected_details)
    report["geometry_failure_count"] = len(failures)
    report["geometry_failures"] = failures
    report["duplicate_coordinate_series"] = sum(
        bool(item["duplicate_physical_coordinates"]) for item in selected_details
    )
    report["slice_count"] = _percentiles(slice_counts)
    report["adjacent_physical_spacing"] = _percentiles(spacings)
    report["physical_extent"] = _percentiles(extents)
    report["selected_series"] = selected_details
    return report


def main() -> int:
    """I run the real-data validation and write its deterministic report."""

    arguments = _parse_args()
    if arguments.dataset_root is None:
        print("REAL-DATA VALIDATION SKIPPED: set --dataset-root or KNEESCOPE_DATA_ROOT")
        return 0
    dataset_root = arguments.dataset_root.expanduser().resolve()
    try:
        report = validate(dataset_root, arguments.series_csv)
    except (FileNotFoundError, ValueError) as error:
        print(f"REAL-DATA VALIDATION SKIPPED: {error}")
        return 0
    if arguments.output is not None:
        output = arguments.output.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return (
        0
        if not report["geometry_failures"] and report["sample_size_selected"] == SAMPLE_SIZE
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())