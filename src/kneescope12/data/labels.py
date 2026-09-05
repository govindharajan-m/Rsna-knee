"""I inspect explicit training-label tables without inventing missing labels."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

_STUDY_ID_COLUMNS = {"study_id", "studyid", "studyinstanceuid", "study_instance_uid"}


@dataclass(frozen=True)
class LabelColumnAudit:
    """I describe observed values, missingness, and encoding for one label column."""

    name: str
    non_missing_count: int
    missing_count: int
    value_counts: dict[str, int]
    binary_encoding_observed: bool


@dataclass(frozen=True)
class LabelAudit:
    """I retain observed label-table facts and unresolved identifier mapping issues."""

    source_path: str
    row_count: int
    study_id_column: str | None
    label_columns: tuple[LabelColumnAudit, ...]

    def to_dict(self) -> dict[str, Any]:
        """I return a JSON-ready audit record."""

        return asdict(self)


def discover_tabular_resources(data_root: str | Path) -> list[Path]:
    """I list CSV and Parquet resources for explicit user review and configuration."""

    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Data root is not a directory: {root}")
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".csv", ".parquet"}
    )


def audit_labels(path: str | Path, *, study_id_column: str | None = None) -> tuple[LabelAudit, list[dict[str, Any]]]:
    """I inspect an explicit label table and leave its provided values unchanged."""

    source = Path(path)
    rows = read_table(source)
    columns = _column_names(rows)
    identifier = study_id_column or _find_study_identifier(columns)
    if study_id_column is not None and study_id_column not in columns:
        raise ValueError(f"Configured study ID column does not exist: {study_id_column}")
    label_columns = [column for column in columns if column != identifier]
    audits = tuple(_audit_column(rows, column) for column in label_columns)
    return (
        LabelAudit(
            source_path=str(source),
            row_count=len(rows),
            study_id_column=identifier,
            label_columns=audits,
        ),
        rows,
    )


def write_clean_label_table(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """I write supplied label-table values to Parquet without imputation or coercion."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), destination, compression="zstd")
    return destination


def read_table(path: str | Path) -> list[dict[str, Any]]:
    """I read a small tabular resource from CSV or Parquet for resource auditing."""

    source = Path(path)
    if source.suffix.lower() == ".csv":
        with source.open("r", encoding="utf-8-sig", newline="") as input_file:
            return [dict(row) for row in csv.DictReader(input_file)]
    if source.suffix.lower() == ".parquet":
        return pq.read_table(source).to_pylist()
    raise ValueError(f"I only support CSV and Parquet label tables: {source}")


def _column_names(rows: list[dict[str, Any]]) -> list[str]:
    """I collect the union of observed columns without presuming a fixed schema."""

    return sorted({column for row in rows for column in row})


def _find_study_identifier(columns: list[str]) -> str | None:
    """I detect only conventional study-ID names and otherwise leave matching unresolved."""

    for column in columns:
        normalized = column.lower().replace(" ", "_")
        if normalized in _STUDY_ID_COLUMNS:
            return column
    return None


def _audit_column(rows: list[dict[str, Any]], column: str) -> LabelColumnAudit:
    """I count raw label encodings and missing values without treating values as truth labels."""

    values = [row.get(column) for row in rows]
    observed = [value for value in values if value not in {None, ""}]
    counts = Counter(str(value) for value in observed)
    binary_values = {str(value).strip().lower() for value in observed}
    return LabelColumnAudit(
        name=column,
        non_missing_count=len(observed),
        missing_count=len(values) - len(observed),
        value_counts=dict(sorted(counts.items())),
        binary_encoding_observed=bool(binary_values) and binary_values <= {"0", "1", "false", "true"},
    )
