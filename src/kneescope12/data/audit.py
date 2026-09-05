"""I scan configurable DICOM datasets without loading image pixel arrays."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kneescope12.utils.config import ConfigError, load_config, resolve_path, resolve_runtime_paths
from kneescope12.utils.logging import configure_logging, get_logger

from .dicom import DicomMetadata, read_dicom_metadata

logger = get_logger("audit")


@dataclass(frozen=True)
class DatasetAuditConfig:
    """I hold explicit scanner rules instead of assuming a dataset layout."""

    scan_all_files: bool = False
    dicom_extensions: tuple[str, ...] = (".dcm", ".dicom", ".ima", "")
    expected_dimensions: tuple[int, int] | None = None
    pixel_spacing_range_mm: tuple[float, float] | None = None
    max_error_samples: int = 100

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> DatasetAuditConfig:
        """I validate a YAML audit section and construct scanner settings."""

        values = values or {}
        if not isinstance(values, Mapping):
            raise ConfigError("audit configuration must be a mapping")
        extensions = values.get("dicom_extensions", cls.dicom_extensions)
        if not isinstance(extensions, Sequence) or isinstance(extensions, (str, bytes)):
            raise ConfigError("audit.dicom_extensions must be a sequence of strings")
        if not all(isinstance(item, str) for item in extensions):
            raise ConfigError("audit.dicom_extensions must be a sequence of strings")
        normalized_extensions = tuple(item.lower() for item in extensions)
        return cls(
            scan_all_files=_optional_bool(
                values.get("scan_all_files", False), "audit.scan_all_files"
            ),
            dicom_extensions=normalized_extensions,
            expected_dimensions=_two_ints(
                values.get("expected_dimensions"), "audit.expected_dimensions"
            ),
            pixel_spacing_range_mm=_spacing_range(values.get("pixel_spacing_range_mm")),
            max_error_samples=_positive_int(
                values.get("max_error_samples", 100), "audit.max_error_samples"
            ),
        )

    def is_candidate(self, path: Path) -> bool:
        """I determine whether a file merits a DICOM parse attempt."""

        return self.scan_all_files or path.suffix.lower() in self.dicom_extensions


@dataclass
class _SeriesAccumulator:
    """I incrementally retain only metadata summaries for one observed series."""

    study_instance_uid: str | None
    series_instance_uid: str | None
    first_relative_path: str
    file_count: int = 0
    descriptions: Counter[str] = field(default_factory=Counter)
    protocols: Counter[str] = field(default_factory=Counter)
    dimensions: Counter[tuple[int, int]] = field(default_factory=Counter)
    pixel_spacings: Counter[tuple[float, float]] = field(default_factory=Counter)
    slice_spacings: Counter[float] = field(default_factory=Counter)
    orientations: Counter[tuple[float, float, float, float, float, float]] = field(
        default_factory=Counter
    )
    lateralities: Counter[str] = field(default_factory=Counter)
    modalities: Counter[str] = field(default_factory=Counter)
    manufacturers: Counter[str] = field(default_factory=Counter)
    manufacturer_models: Counter[str] = field(default_factory=Counter)
    sequence_names: Counter[str] = field(default_factory=Counter)
    acquisition_dates: Counter[str] = field(default_factory=Counter)
    acquisition_times: Counter[str] = field(default_factory=Counter)
    magnetic_field_strengths: Counter[float] = field(default_factory=Counter)
    missing: Counter[str] = field(default_factory=Counter)
    positions: set[tuple[float, float, float]] = field(default_factory=set)
    duplicate_positions: int = 0

    def add(self, metadata: DicomMetadata) -> None:
        """I consume one metadata object and update compact per-series counters."""

        self.file_count += 1
        _count_or_missing(
            self.descriptions, self.missing, "series_description", metadata.series_description
        )
        _count_or_missing(self.protocols, self.missing, "protocol_name", metadata.protocol_name)
        _count_or_missing(self.dimensions, self.missing, "image_dimensions", metadata.dimensions)
        _count_or_missing(
            self.pixel_spacings, self.missing, "pixel_spacing", metadata.pixel_spacing
        )
        _count_or_missing(
            self.orientations,
            self.missing,
            "image_orientation_patient",
            metadata.image_orientation_patient,
        )
        _count_or_missing(self.lateralities, self.missing, "laterality", metadata.laterality)
        _count_or_missing(self.modalities, self.missing, "modality", metadata.modality)
        _count_or_missing(self.manufacturers, self.missing, "manufacturer", metadata.manufacturer)
        _count_or_missing(
            self.manufacturer_models,
            self.missing,
            "manufacturer_model_name",
            metadata.manufacturer_model_name,
        )
        _count_or_missing(
            self.sequence_names, self.missing, "sequence_name", metadata.sequence_name
        )
        _count_or_missing(
            self.acquisition_dates, self.missing, "acquisition_date", metadata.acquisition_date
        )
        _count_or_missing(
            self.acquisition_times, self.missing, "acquisition_time", metadata.acquisition_time
        )
        _count_or_missing(
            self.magnetic_field_strengths,
            self.missing,
            "magnetic_field_strength",
            metadata.magnetic_field_strength,
        )
        if metadata.spacing_between_slices is not None:
            self.slice_spacings[metadata.spacing_between_slices] += 1
        elif metadata.slice_thickness is not None:
            self.slice_spacings[metadata.slice_thickness] += 1
        else:
            self.missing["slice_spacing"] += 1
        if metadata.image_position_patient is None:
            self.missing["image_position_patient"] += 1
        elif metadata.image_position_patient in self.positions:
            self.duplicate_positions += 1
        else:
            self.positions.add(metadata.image_position_patient)

    def qc_flags(self, config: DatasetAuditConfig) -> list[str]:
        """I derive conservative QC flags from observed metadata differences."""

        flags: list[str] = []
        if self.study_instance_uid is None:
            flags.append("missing_study_instance_uid")
        if self.series_instance_uid is None:
            flags.append("missing_series_instance_uid")
        flags.extend(_missing_flag(field) for field in sorted(self.missing))
        if len(self.dimensions) > 1:
            flags.append("inconsistent_image_dimensions")
        if len(self.pixel_spacings) > 1:
            flags.append("inconsistent_pixel_spacing")
        if len(self.orientations) > 1:
            flags.append("inconsistent_orientation")
        if len(self.slice_spacings) > 1:
            flags.append("inconsistent_slice_spacing")
        if self.duplicate_positions:
            flags.append("duplicate_slice_position")
        if config.expected_dimensions is not None and any(
            dimensions != config.expected_dimensions for dimensions in self.dimensions
        ):
            flags.append("unexpected_image_dimensions")
        if config.pixel_spacing_range_mm is not None:
            low, high = config.pixel_spacing_range_mm
            if any(
                value < low or value > high for spacing in self.pixel_spacings for value in spacing
            ):
                flags.append("unusual_pixel_spacing")
        return flags

    def to_row(self, config: DatasetAuditConfig) -> dict[str, Any]:
        """I create one serializable inventory row for this observed series."""

        return {
            "study_instance_uid": self.study_instance_uid,
            "series_instance_uid": self.series_instance_uid,
            "first_relative_path": self.first_relative_path,
            "dicom_file_count": self.file_count,
            "series_descriptions": _format_counter(self.descriptions),
            "protocol_names": _format_counter(self.protocols),
            "modalities": _format_counter(self.modalities),
            "image_dimensions": _format_dimensions(self.dimensions),
            "pixel_spacings_mm": _format_float_tuples(self.pixel_spacings),
            "slice_spacings_mm": _format_float_counter(self.slice_spacings),
            "orientation_count": len(self.orientations),
            "lateralities": _format_counter(self.lateralities),
            "manufacturers": _format_counter(self.manufacturers),
            "manufacturer_models": _format_counter(self.manufacturer_models),
            "sequence_names": _format_counter(self.sequence_names),
            "acquisition_dates": _format_counter(self.acquisition_dates),
            "acquisition_times": _format_counter(self.acquisition_times),
            "magnetic_field_strengths_t": _format_float_counter(self.magnetic_field_strengths),
            "missing_metadata": _format_counter(self.missing),
            "qc_flags": "; ".join(self.qc_flags(config)),
        }


@dataclass(frozen=True)
class DatasetAuditResult:
    """I retain audit counts, errors, and series inventory rows for output."""

    dataset_root: Path
    total_files_seen: int
    candidate_files: int
    readable_dicom_files: int
    unreadable_candidate_files: int
    study_count: int
    series_rows: tuple[dict[str, Any], ...]
    series_description_counts: dict[str, int]
    image_dimension_counts: dict[str, int]
    pixel_spacing_counts: dict[str, int]
    qc_flag_counts: dict[str, int]
    study_qc_flags: dict[str, list[str]]
    unreadable_samples: tuple[dict[str, str], ...]

    @property
    def series_count(self) -> int:
        """I return the number of observed DICOM series groups."""

        return len(self.series_rows)

    def to_summary_dict(self) -> dict[str, Any]:
        """I create a machine-readable audit summary without raw image data."""

        return {
            "dataset_root": str(self.dataset_root),
            "counts": {
                "files_seen": self.total_files_seen,
                "candidate_files": self.candidate_files,
                "readable_dicom_files": self.readable_dicom_files,
                "unreadable_candidate_files": self.unreadable_candidate_files,
                "studies_with_uid": self.study_count,
                "series": self.series_count,
            },
            "series_description_counts": self.series_description_counts,
            "image_dimension_counts": self.image_dimension_counts,
            "pixel_spacing_counts_mm": self.pixel_spacing_counts,
            "qc_flag_counts": self.qc_flag_counts,
            "study_qc_flags": self.study_qc_flags,
            "unreadable_file_samples": list(self.unreadable_samples),
        }

    def human_summary(self) -> str:
        """I provide a compact console summary that points to structured outputs."""

        counts = self.to_summary_dict()["counts"]
        flagged_series_count = sum(bool(row["qc_flags"]) for row in self.series_rows)
        return "\n".join(
            [
                "Dataset audit summary",
                f"dataset root: {self.dataset_root}",
                f"studies with DICOM StudyInstanceUID: {counts['studies_with_uid']}",
                f"series: {counts['series']}",
                f"readable DICOM files: {counts['readable_dicom_files']}",
                f"unreadable candidate files: {counts['unreadable_candidate_files']}",
                f"series carrying QC flags: {flagged_series_count}",
            ]
        )


class DatasetAuditor:
    """I scan filesystem candidates one at a time and summarize DICOM metadata."""

    def __init__(self, config: DatasetAuditConfig | None = None) -> None:
        """I initialize the auditor with explicit, conservative scan settings."""

        self.config = config or DatasetAuditConfig()

    def scan(self, dataset_root: str | Path, *, limit: int | None = None) -> DatasetAuditResult:
        """I scan a directory lazily and return metadata-only audit results."""

        root = Path(dataset_root)
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset root is not a directory: {root}")
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive when supplied")

        series: dict[tuple[str, str], _SeriesAccumulator] = {}
        study_uids: set[str] = set()
        unreadable_samples: list[dict[str, str]] = []
        total_files_seen = candidate_files = readable_dicom_files = unreadable_candidate_files = 0

        for file_path in _iter_files(root):
            total_files_seen += 1
            if not self.config.is_candidate(file_path):
                continue
            candidate_files += 1
            read_result = read_dicom_metadata(file_path)
            if not read_result.succeeded:
                unreadable_candidate_files += 1
                if len(unreadable_samples) < self.config.max_error_samples:
                    unreadable_samples.append(
                        {
                            "path": str(file_path.relative_to(root)),
                            "error": read_result.error or "Unknown DICOM read failure",
                        }
                    )
                continue

            metadata = read_result.metadata
            if metadata is None:
                raise RuntimeError("I received a successful DICOM result without metadata")
            readable_dicom_files += 1
            if metadata.study_instance_uid is not None:
                study_uids.add(metadata.study_instance_uid)
            key = _series_key(metadata, file_path.relative_to(root))
            accumulator = series.get(key)
            if accumulator is None:
                accumulator = _SeriesAccumulator(
                    study_instance_uid=metadata.study_instance_uid,
                    series_instance_uid=metadata.series_instance_uid,
                    first_relative_path=str(file_path.relative_to(root)),
                )
                series[key] = accumulator
            accumulator.add(metadata)
            if limit is not None and len(series) >= limit:
                break

        series_rows = tuple(
            accumulator.to_row(self.config)
            for _, accumulator in sorted(series.items(), key=lambda item: item[0])
        )
        _append_duplicate_looking_series_flags(series_rows)
        return DatasetAuditResult(
            dataset_root=root.resolve(),
            total_files_seen=total_files_seen,
            candidate_files=candidate_files,
            readable_dicom_files=readable_dicom_files,
            unreadable_candidate_files=unreadable_candidate_files,
            study_count=len(study_uids),
            series_rows=series_rows,
            series_description_counts=_aggregate_counter(series_rows, "series_descriptions"),
            image_dimension_counts=_aggregate_counter(series_rows, "image_dimensions"),
            pixel_spacing_counts=_aggregate_counter(series_rows, "pixel_spacings_mm"),
            qc_flag_counts=_count_qc_flags(series_rows),
            study_qc_flags=_study_qc_flags(series_rows),
            unreadable_samples=tuple(unreadable_samples),
        )


def write_audit_outputs(result: DatasetAuditResult, output_dir: str | Path) -> dict[str, Path]:
    """I write human-readable and machine-readable outputs for an audit result."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary_path = destination / "audit_summary.json"
    inventory_path = destination / "series_inventory.csv"
    study_inventory_path = destination / "study_inventory.csv"
    report_path = destination / "audit_report.md"
    summary_path.write_text(
        json.dumps(result.to_summary_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fieldnames = list(_inventory_fieldnames())
    with inventory_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result.series_rows)
    with study_inventory_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["study_instance_uid", "series_count", "series_descriptions", "qc_flags"],
        )
        writer.writeheader()
        writer.writerows(_study_inventory_rows(result.series_rows))
    report_path.write_text(_human_audit_report(result), encoding="utf-8")
    logger.info("I wrote audit summary to %s", summary_path)
    logger.info("I wrote series inventory to %s", inventory_path)
    logger.info("I wrote study inventory to %s", study_inventory_path)
    logger.info("I wrote human-readable audit report to %s", report_path)
    return {
        "summary": summary_path,
        "inventory": inventory_path,
        "study_inventory": study_inventory_path,
        "report": report_path,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    """I build the standalone audit command interface."""

    parser = argparse.ArgumentParser(
        description="Audit DICOM metadata without loading image pixels."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_project_root() / "configs" / "baseline.yaml",
        help="YAML configuration file (default: configs/baseline.yaml).",
    )
    parser.add_argument(
        "--data-root",
        "--dataset-root",
        dest="dataset_root",
        type=Path,
        help="Dataset directory to audit.",
    )
    parser.add_argument("--output-dir", type=Path, help="Directory for JSON, CSV, and log outputs.")
    parser.add_argument("--log-level", help="Logging level that overrides the YAML setting.")
    parser.add_argument(
        "--scan-all-files",
        action="store_true",
        help="Attempt DICOM parsing for every regular file, including unusual extensions.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after this many series are discovered during the audit.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """I run the dataset-audit command and return a shell-compatible status."""

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        config = resolve_runtime_paths(load_config(args.config), project_root=_project_root())
        path_values = config.get("paths", {})
        if not isinstance(path_values, Mapping):
            raise ConfigError("paths configuration must be a mapping")
        root_value = args.dataset_root or path_values.get("dataset_root")
        if root_value is None:
            parser.error("I need --data-root or paths.dataset_root in the configuration.")
        output_value = args.output_dir or path_values.get(
            "audit_output", "experiments/dataset_audit"
        )
        dataset_root = resolve_path(root_value, project_root=_project_root())
        output_dir = resolve_path(output_value, project_root=_project_root())
        audit_config = DatasetAuditConfig.from_mapping(config.get("audit"))
        if args.scan_all_files:
            audit_config = DatasetAuditConfig(
                scan_all_files=True,
                dicom_extensions=audit_config.dicom_extensions,
                expected_dimensions=audit_config.expected_dimensions,
                pixel_spacing_range_mm=audit_config.pixel_spacing_range_mm,
                max_error_samples=audit_config.max_error_samples,
            )
        logging_values = config.get("logging", {})
        if not isinstance(logging_values, Mapping):
            raise ConfigError("logging configuration must be a mapping")
        level = args.log_level or logging_values.get("level", "INFO")
        configure_logging(level, log_file=output_dir / "audit.log")
        result = DatasetAuditor(audit_config).scan(dataset_root, limit=args.limit)
        output_paths = write_audit_outputs(result, output_dir)
    except (ConfigError, FileNotFoundError, OSError, ValueError) as error:
        parser.error(str(error))

    logger.info("%s", result.human_summary())
    logger.info("I saved structured outputs under %s", output_paths["summary"].parent)
    return 0


def _iter_files(root: Path) -> Iterable[Path]:
    """I yield regular files a directory at a time instead of preloading a file list."""

    for directory, directories, filenames in os.walk(root):
        directories.sort()
        filenames.sort()
        current = Path(directory)
        for filename in filenames:
            candidate = current / filename
            if candidate.is_file():
                yield candidate


def _series_key(metadata: DicomMetadata, relative_path: Path) -> tuple[str, str]:
    """I use UIDs when present and retain an explicit path fallback when they are absent."""

    study_key = metadata.study_instance_uid or "<missing-study-uid>"
    series_key = metadata.series_instance_uid or f"<missing-series-uid:{relative_path}>"
    return study_key, series_key


def _append_duplicate_looking_series_flags(rows: tuple[dict[str, Any], ...]) -> None:
    """I flag series sharing a conservative metadata signature within an identified study."""

    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        study_uid = row["study_instance_uid"]
        if study_uid is None:
            continue
        signature = (
            study_uid,
            row["series_descriptions"],
            row["image_dimensions"],
            str(row["orientation_count"]),
        )
        groups[signature].append(row)
    for grouped_rows in groups.values():
        if len(grouped_rows) > 1:
            for row in grouped_rows:
                flags = [flag for flag in row["qc_flags"].split("; ") if flag]
                row["qc_flags"] = "; ".join(sorted(set([*flags, "duplicate_looking_series"])))


def _aggregate_counter(rows: tuple[dict[str, Any], ...], column: str) -> dict[str, int]:
    """I aggregate compact counter strings used in the CSV inventory."""

    counts: Counter[str] = Counter()
    for row in rows:
        for entry in str(row[column]).split("; "):
            if not entry:
                continue
            name, separator, count = entry.rpartition(" (")
            if not separator or not count.endswith(")"):
                continue
            try:
                counts[name] += int(count[:-1])
            except ValueError:
                logger.warning("I could not aggregate malformed inventory entry: %s", entry)
    return dict(sorted(counts.items()))


def _count_qc_flags(rows: tuple[dict[str, Any], ...]) -> dict[str, int]:
    """I count QC flags across the series inventory."""

    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(flag for flag in row["qc_flags"].split("; ") if flag)
    return dict(sorted(counts.items()))


def _study_qc_flags(rows: tuple[dict[str, Any], ...]) -> dict[str, list[str]]:
    """I combine series QC flags into study-level views where a study UID exists."""

    flags_by_study: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        study_uid = row["study_instance_uid"]
        if study_uid is not None:
            flags_by_study[study_uid].update(flag for flag in row["qc_flags"].split("; ") if flag)
    return {study: sorted(flags) for study, flags in sorted(flags_by_study.items()) if flags}


def _count_or_missing(
    counter: Counter[Any], missing: Counter[str], name: str, value: Any | None
) -> None:
    """I update a value counter or record one missing metadata field."""

    if value is None:
        missing[name] += 1
    else:
        counter[value] += 1


def _format_counter(counter: Counter[Any]) -> str:
    """I turn text counters into stable CSV cells."""

    return "; ".join(
        f"{key} ({count})" for key, count in sorted(counter.items(), key=lambda item: str(item[0]))
    )


def _format_dimensions(counter: Counter[tuple[int, int]]) -> str:
    """I turn image-dimension counters into stable CSV cells."""

    return "; ".join(
        f"{rows}x{columns} ({count})" for (rows, columns), count in sorted(counter.items())
    )


def _format_float_tuples(counter: Counter[tuple[float, float]]) -> str:
    """I turn pixel-spacing counters into stable CSV cells."""

    return "; ".join(
        f"{row_spacing:g}x{column_spacing:g} ({count})"
        for (row_spacing, column_spacing), count in sorted(counter.items())
    )


def _format_float_counter(counter: Counter[float]) -> str:
    """I turn scalar spacing counters into stable CSV cells."""

    return "; ".join(f"{value:g} ({count})" for value, count in sorted(counter.items()))


def _missing_flag(name: str) -> str:
    """I make a stable QC flag name for a missing metadata field."""

    return f"missing_{name}"


def _inventory_fieldnames() -> tuple[str, ...]:
    """I define the ordered CSV schema for a series inventory."""

    return (
        "study_instance_uid",
        "series_instance_uid",
        "first_relative_path",
        "dicom_file_count",
        "series_descriptions",
        "protocol_names",
        "modalities",
        "image_dimensions",
        "pixel_spacings_mm",
        "slice_spacings_mm",
        "orientation_count",
        "lateralities",
        "manufacturers",
        "manufacturer_models",
        "sequence_names",
        "acquisition_dates",
        "acquisition_times",
        "magnetic_field_strengths_t",
        "missing_metadata",
        "qc_flags",
    )


def _study_inventory_rows(rows: tuple[dict[str, Any], ...]) -> list[dict[str, str | int | None]]:
    """I summarize observed series under each study UID without choosing a preferred series."""

    grouped: dict[str | None, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["study_instance_uid"]].append(row)
    inventory: list[dict[str, str | int | None]] = []
    for study_id, study_rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        descriptions = sorted(
            {description for row in study_rows for description in row["series_descriptions"].split("; ")}
        )
        flags = sorted({flag for row in study_rows for flag in row["qc_flags"].split("; ") if flag})
        inventory.append(
            {
                "study_instance_uid": study_id,
                "series_count": len(study_rows),
                "series_descriptions": "; ".join(filter(None, descriptions)),
                "qc_flags": "; ".join(flags),
            }
        )
    return inventory


def _human_audit_report(result: DatasetAuditResult) -> str:
    """I render observed audit counts as a concise Markdown review document."""

    counts = result.to_summary_dict()["counts"]
    lines = [
        "# Dataset Audit Report",
        "",
        "I generated this report from the configured data root without loading image pixels.",
        "",
        "## Observed counts",
        "",
        f"- Studies with DICOM StudyInstanceUID: {counts['studies_with_uid']}",
        f"- Observed series: {counts['series']}",
        f"- Readable DICOM files: {counts['readable_dicom_files']}",
        f"- Unreadable candidate files: {counts['unreadable_candidate_files']}",
        "",
        "## Series-description prevalence",
        "",
    ]
    if result.series_description_counts:
        lines.extend(
            f"- {description}: {count}" for description, count in result.series_description_counts.items()
        )
    else:
        lines.append("- No readable DICOM series descriptions were observed.")
    lines.extend(["", "## Quality-control flags", ""])
    if result.qc_flag_counts:
        lines.extend(f"- {flag}: {count}" for flag, count in result.qc_flag_counts.items())
    else:
        lines.append("- No series-level flags were produced.")
    lines.extend(
        [
            "",
            "I treat this report as a measured inventory, not a decision about which MRI sequences to use.",
            "I inspect slice ordering and reconstructed volumes separately before relying on any series.",
            "",
        ]
    )
    return "\n".join(lines)


def _optional_bool(value: Any, field_name: str) -> bool:
    """I validate an explicit boolean setting."""

    if not isinstance(value, bool):
        raise ConfigError(f"{field_name} must be true or false")
    return value


def _two_ints(value: Any, field_name: str) -> tuple[int, int] | None:
    """I validate an optional two-integer setting."""

    if value is None:
        return None
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
        or not all(isinstance(item, int) and item > 0 for item in value)
    ):
        raise ConfigError(f"{field_name} must be null or two positive integers")
    return int(value[0]), int(value[1])


def _spacing_range(value: Any) -> tuple[float, float] | None:
    """I validate an optional inclusive pixel-spacing range in millimetres."""

    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ConfigError("audit.pixel_spacing_range_mm must be null or two numbers")
    try:
        lower, upper = float(value[0]), float(value[1])
    except (TypeError, ValueError) as error:
        raise ConfigError("audit.pixel_spacing_range_mm must be null or two numbers") from error
    if lower <= 0 or upper < lower:
        raise ConfigError("audit.pixel_spacing_range_mm must have 0 < lower <= upper")
    return lower, upper


def _positive_int(value: Any, field_name: str) -> int:
    """I validate a positive integer setting."""

    if not isinstance(value, int) or value < 1:
        raise ConfigError(f"{field_name} must be a positive integer")
    return value


def _project_root() -> Path:
    """I locate this repository relative to the installed source tree."""

    return Path(__file__).resolve().parents[3]


if __name__ == "__main__":
    raise SystemExit(main())
