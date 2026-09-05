"""I test metadata-only dataset scanning against synthetic directory contents."""

from __future__ import annotations

import json

from kneescope12.data.audit import DatasetAuditConfig, DatasetAuditor, write_audit_outputs
from kneescope12.data.dicom import read_dicom_metadata


def test_dataset_auditor_reports_dicom_and_malformed_candidates(tmp_path, write_dicom):
    """I scan DICOM files incrementally and retain malformed-file evidence."""

    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    first = write_dicom(dataset_root / "first.dcm", position_z=0.0)
    first_result = read_dicom_metadata(first)
    assert first_result.metadata is not None
    study_uid = first_result.metadata.study_instance_uid
    series_uid = first_result.metadata.series_instance_uid
    write_dicom(
        dataset_root / "second.dcm",
        study_uid=study_uid,
        series_uid=series_uid,
        instance_number=2,
        position_z=3.0,
    )
    (dataset_root / "broken.dcm").write_text("broken", encoding="utf-8")
    (dataset_root / "notes.txt").write_text("not scanned by default", encoding="utf-8")

    result = DatasetAuditor(DatasetAuditConfig()).scan(dataset_root)
    assert result.total_files_seen == 4
    assert result.candidate_files == 3
    assert result.readable_dicom_files == 2
    assert result.unreadable_candidate_files == 1
    assert result.study_count == 1
    assert result.series_count == 1
    assert result.series_rows[0]["dicom_file_count"] == 2

    output_paths = write_audit_outputs(result, tmp_path / "audit-output")
    summary = json.loads(output_paths["summary"].read_text(encoding="utf-8"))
    assert summary["counts"]["unreadable_candidate_files"] == 1
    assert output_paths["inventory"].is_file()
