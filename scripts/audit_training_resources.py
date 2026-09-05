"""I inspect supplied label and report resources without training or storing report text."""

from __future__ import annotations

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

labels_module = import_module("kneescope12.data.labels")
reports_module = import_module("kneescope12.data.reports")


def main() -> int:
    """I create non-sensitive label and report audit outputs from an explicit data root."""

    parser = argparse.ArgumentParser(description="Audit explicit label and report resources.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--labels-path", type=Path)
    parser.add_argument("--study-id-column")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "experiments" / "resource_audit")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_candidates = labels_module.discover_tabular_resources(args.data_root)
    selected_labels = args.labels_path
    label_audit = None
    known_study_ids = None
    if selected_labels is not None:
        label_audit, label_rows = labels_module.audit_labels(
            selected_labels, study_id_column=args.study_id_column
        )
        labels_module.write_clean_label_table(label_rows, args.output_dir / "labels.parquet")
        if label_audit.study_id_column is not None:
            known_study_ids = {
                str(row[label_audit.study_id_column])
                for row in label_rows
                if row.get(label_audit.study_id_column) not in {None, ""}
            }
    report_audit, report_rows = reports_module.audit_reports(
        args.data_root, known_study_ids=known_study_ids
    )
    reports_module.write_report_manifest(report_rows, args.output_dir / "report_manifest.parquet")
    payload = {
        "tabular_candidates": [str(path) for path in table_candidates],
        "selected_labels_path": str(selected_labels) if selected_labels is not None else None,
        "label_audit": label_audit.to_dict() if label_audit is not None else None,
        "report_audit": report_audit.to_dict(),
    }
    (args.output_dir / "resource_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote resource audit to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
