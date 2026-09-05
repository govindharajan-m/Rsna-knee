"""I audit provided radiology-report resources without placing report text in logs or manifests."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException
import pyarrow as pa
import pyarrow.parquet as pq

from .labels import _find_study_identifier, read_table

DetectorFactory.seed = 0
_REPORT_COLUMNS = {"report", "report_text", "text", "impression", "findings"}


@dataclass(frozen=True)
class ReportManifestRow:
    """I expose matching and length metadata without retaining report text."""

    study_id: str | None
    source_path: str
    source_format: str
    character_count: int
    language: str
    match_status: str


@dataclass(frozen=True)
class ReportAudit:
    """I summarize report availability and matching without logging clinical content."""

    candidate_files: tuple[str, ...]
    report_count: int
    matched_study_count: int
    missing_study_count: int
    language_counts: dict[str, int]
    format_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """I return a JSON-ready, text-free report audit record."""

        return asdict(self)


def audit_reports(
    data_root: str | Path,
    *,
    known_study_ids: set[str] | None = None,
) -> tuple[ReportAudit, list[ReportManifestRow]]:
    """I inspect text and tabular report candidates with explicit matching evidence."""

    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Data root is not a directory: {root}")
    rows: list[ReportManifestRow] = []
    candidates: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() == ".txt":
            candidates.append(path)
            rows.append(_row_from_text(path, None, known_study_ids))
        elif path.suffix.lower() in {".csv", ".parquet"}:
            table_rows = read_table(path)
            if _report_column(table_rows) is not None:
                candidates.append(path)
                rows.extend(_rows_from_table(path, table_rows, known_study_ids))
    languages = Counter(row.language for row in rows)
    formats = Counter(row.source_format for row in rows)
    matched = sum(row.match_status == "matched" for row in rows)
    return (
        ReportAudit(
            candidate_files=tuple(str(path) for path in candidates),
            report_count=len(rows),
            matched_study_count=matched,
            missing_study_count=sum(row.match_status != "matched" for row in rows),
            language_counts=dict(sorted(languages.items())),
            format_counts=dict(sorted(formats.items())),
        ),
        rows,
    )


def write_report_manifest(rows: list[ReportManifestRow], path: str | Path) -> Path:
    """I write text-free report metadata to Parquet for later matching analysis."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([asdict(row) for row in rows]), destination, compression="zstd")
    return destination


def _rows_from_table(
    path: Path,
    table_rows: list[dict[str, Any]],
    known_study_ids: set[str] | None,
) -> list[ReportManifestRow]:
    """I derive text-free report manifest rows when a table exposes a report column."""

    report_column = _report_column(table_rows)
    if report_column is None:
        return []
    identifier = _find_study_identifier(sorted({key for row in table_rows for key in row}))
    records: list[ReportManifestRow] = []
    for row in table_rows:
        text = row.get(report_column)
        if text in {None, ""}:
            continue
        study_id = str(row[identifier]) if identifier is not None and row.get(identifier) is not None else None
        records.append(_manifest_row(study_id, str(path), path.suffix.lower().lstrip("."), str(text), known_study_ids))
    return records


def _row_from_text(
    path: Path,
    study_id: str | None,
    known_study_ids: set[str] | None,
) -> ReportManifestRow:
    """I inspect a standalone report without assuming its filename maps to a study."""

    text = path.read_text(encoding="utf-8", errors="replace")
    return _manifest_row(study_id, str(path), "txt", text, known_study_ids)


def _manifest_row(
    study_id: str | None,
    source_path: str,
    source_format: str,
    text: str,
    known_study_ids: set[str] | None,
) -> ReportManifestRow:
    """I compute non-sensitive report metadata and an explicit match status."""

    if known_study_ids is None:
        match_status = "not_evaluated"
    elif study_id is None:
        match_status = "missing_study_identifier"
    elif study_id in known_study_ids:
        match_status = "matched"
    else:
        match_status = "unmatched_study_identifier"
    return ReportManifestRow(
        study_id=study_id,
        source_path=source_path,
        source_format=source_format,
        character_count=len(text),
        language=_detect_language(text),
        match_status=match_status,
    )


def _report_column(rows: list[dict[str, Any]]) -> str | None:
    """I detect only conventional report-text field names in a supplied table."""

    columns = sorted({key for row in rows for key in row})
    for column in columns:
        if column.lower().replace(" ", "_") in _REPORT_COLUMNS:
            return column
    return None


def _detect_language(text: str) -> str:
    """I return an estimated language code or undetermined when text is insufficient."""

    if len(text.strip()) < 20:
        return "undetermined"
    try:
        return detect(text)
    except LangDetectException:
        return "undetermined"
