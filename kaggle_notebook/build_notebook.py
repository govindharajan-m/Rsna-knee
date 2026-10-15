"""I build a self-contained Kaggle notebook from the checked-in baseline sources."""

from __future__ import annotations

import base64
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = Path(__file__).with_name("knee12.ipynb")
KERNEL_ID = "govind123445/knee-12"
EMBEDDED_FILES = (
    "configs/baseline.yaml",
    "src/kneescope12/__init__.py",
    "src/kneescope12/data/__init__.py",
    "src/kneescope12/data/dataset.py",
    "src/kneescope12/data/dicom.py",
    "src/kneescope12/data/intensity.py",
    "src/kneescope12/data/normalization.py",
    "src/kneescope12/data/ordering.py",
    "src/kneescope12/data/preprocessing.py",
    "src/kneescope12/data/resampling.py",
    "src/kneescope12/data/roi.py",
    "src/kneescope12/data/volume.py",
    "src/kneescope12/model/__init__.py",
    "src/kneescope12/model/backbone.py",
    "src/kneescope12/model/classifier.py",
    "src/kneescope12/utils/__init__.py",
    "src/kneescope12/utils/logging.py",
    "src/kneescope12/utils/reproducibility.py",
    "scripts/predict_baseline.py",
    "scripts/train_baseline.py",
)


def _source_archive() -> str:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for relative in EMBEDDED_FILES:
            output.write(ROOT / relative, relative)
    return base64.b64encode(archive.getvalue()).decode("ascii")


def _cell(cell_type: str, source: str) -> dict[str, object]:
    content = source.splitlines(keepends=True)
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": content,
        **({"outputs": [], "execution_count": None} if cell_type == "code" else {}),
    }


def main() -> None:
    archive_b64 = _source_archive()
    cells = [
        _cell(
            "markdown",
            """# KneeScope-12: MRI-only baseline

I train a small 3D CNN using only studies with all twelve explicit binary labels,
report a deterministic study-level validation split, fit a final model on all
explicitly labeled studies, and generate `/kaggle/working/submission.csv`.

This notebook has internet disabled and embeds the exact project source files
listed by `build_notebook.py`; it does not clone the repository or download
weights. The input competition data is read-only. For each known manifest plane,
the training policy selects the lexicographically first SeriesInstanceUID. If
the test release has no series manifest, I assign test series to planes using a
one-to-one mapping between manifest planes and dominant DICOM slice-normal axes
measured from the selected gold-training series, then choose the first UID in
each inferred plane. This empirical mapping is not a patient-side convention or
an optimal series choice.
""",
        ),
        _cell(
            "code",
            f"""from pathlib import Path
import base64
import csv
import importlib.util
import io
import json
import sys
import zipfile

import numpy as np
import pydicom
import scipy
import torch
import yaml

DATA_ROOT = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
WORK_ROOT = Path("/kaggle/working")
SOURCE_ROOT = WORK_ROOT / "kneescope12_embedded"
SOURCE_ARCHIVE_B64 = {archive_b64!r}

required = [
    DATA_ROOT / "train.csv",
    DATA_ROOT / "train_series.csv",
    DATA_ROOT / "sample_submission.csv",
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise FileNotFoundError(f"Required competition files are missing: {{missing}}")

test_manifest = DATA_ROOT / "test_series.csv"
if not test_manifest.is_file():
    test_manifest = None
test_series_roots = [
    name for name in ("test_series", "test_images") if (DATA_ROOT / name).is_dir()
]
if len(test_series_roots) != 1:
    raise FileNotFoundError(
        f"Expected exactly one test DICOM root (test_series or test_images); "
        f"found {{test_series_roots}}"
    )
test_series_directory = test_series_roots[0]

with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE_ARCHIVE_B64))) as archive:
    archive.extractall(SOURCE_ROOT)
PROJECT_ROOT = SOURCE_ROOT
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))
print(json.dumps({{
    "data_root": str(DATA_ROOT),
    "test_manifest": str(test_manifest),
    "test_series_directory": test_series_directory,
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "pydicom": pydicom.__version__,
    "scipy": scipy.__version__,
}}, indent=2))
""",
        ),
        _cell(
            "code",
            """from kneescope12.data.dataset import read_gold_labels, _read_series_candidates

config_path = PROJECT_ROOT / "configs" / "baseline.yaml"
with config_path.open(encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
label_columns = list(config["labels"])
with (DATA_ROOT / "train.csv").open(newline="", encoding="utf-8-sig") as handle:
    train_reader = csv.DictReader(handle)
    train_columns = train_reader.fieldnames or []
if "StudyInstanceUID" not in train_columns:
    raise ValueError(f"train.csv lacks StudyInstanceUID; columns were {train_columns}")
missing_labels = sorted(set(label_columns) - set(train_columns))
if missing_labels:
    raise ValueError(f"train.csv lacks configured target columns: {missing_labels}")
gold_labels = read_gold_labels(DATA_ROOT / "train.csv", label_columns)
if len(gold_labels) != 58:
    raise ValueError(
        f"Expected the documented 58 fully labeled studies, found {len(gold_labels)}"
    )

sample_path = DATA_ROOT / "sample_submission.csv"
with sample_path.open(newline="", encoding="utf-8-sig") as handle:
    sample_reader = csv.reader(handle)
    sample_columns = next(sample_reader)
    test_study_ids = [row[0].strip() for row in sample_reader if row and row[0].strip()]
if sample_columns[0] != "StudyInstanceUID" or sample_columns[1:] != label_columns:
    raise ValueError(
        "sample_submission.csv target columns/order differ from the configured label order"
    )
test_manifest_series_count = None
if test_manifest is not None:
    with test_manifest.open(newline="", encoding="utf-8-sig") as handle:
        manifest_columns = next(csv.reader(handle))
    required_manifest = {{
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "Fluid_Sensitive",
        "Fat_Suppression",
        "Anatomical_Plane",
    }}
    if not required_manifest.issubset(set(manifest_columns)):
        raise ValueError(
            f"Test series manifest does not expose the RSNA schema: {{manifest_columns}}"
        )
    test_candidates = _read_series_candidates(test_manifest, test_study_ids)
    missing_test_series = [
        study for study in test_study_ids if not test_candidates.get(study)
    ]
    if missing_test_series:
        raise ValueError(
            f"Test manifest has no series rows for {{len(missing_test_series)}} sample studies; "
            f"examples: {{missing_test_series[:5]}}"
        )
    test_manifest_series_count = sum(len(rows) for rows in test_candidates.values())
print(json.dumps({{
    "train_columns": train_columns,
    "manifest_series_count": sum(
        1 for _ in open(DATA_ROOT / "train_series.csv", encoding="utf-8-sig")
    ) - 1,
    "explicit_gold_labeled_studies": len(gold_labels),
    "test_study_count": len(test_study_ids),
    "test_series_manifest_rows_for_sample": test_manifest_series_count,
    "labels": label_columns,
}}, indent=2))
""",
        ),
        _cell(
            "markdown",
            """## Validation and full-label fit

The first fit uses a seeded 80/20 split by StudyInstanceUID, reports ROC AUC
only for labels with both classes in validation, and saves its measured metrics.
The final fit then resets the model and trains on all 58 explicitly labeled
studies for the same number of epochs. Blank report-only rows are never converted
to negatives. No reports are parsed for this baseline.
""",
        ),
        _cell(
            "code",
            """from types import SimpleNamespace
from scripts.train_baseline import run_training

config["training"]["epochs"] = 3
config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
validation_args = SimpleNamespace(
    config=config_path,
    dataset_root=DATA_ROOT,
    labels_csv=DATA_ROOT / "train.csv",
    series_csv=DATA_ROOT / "train_series.csv",
    checkpoint=WORK_ROOT / "baseline_validation.pt",
    seed=int(config["seed"]),
    epochs=3,
    batch_size=int(config["training"]["batch_size"]),
    learning_rate=float(config["training"]["learning_rate"]),
    device="auto",
    max_train_batches=None,
    smoke_test=False,
    train_all=False,
)
validation_result = run_training(validation_args)
print(json.dumps({{
    "train_study_count": validation_result["train_study_count"],
    "validation_study_count": validation_result["validation_study_count"],
    "validation_history": [
        {{"epoch": row["epoch"], "loss": row["mean_train_loss"], "metrics": row["validation"]}}
        for row in validation_result["history"]
    ],
}}, indent=2))

final_args = SimpleNamespace(
    **{{**vars(validation_args),
       "checkpoint": WORK_ROOT / "baseline_final.pt",
       "train_all": True}}
)
final_result = run_training(final_args)
print(json.dumps({{
    "final_fit_studies": final_result["gold_labeled_study_count"],
    "checkpoint": final_result["checkpoint"],
    "epochs": len(final_result["history"]),
}}, indent=2))
""",
        ),
        _cell(
            "markdown",
            """## Test inference and submission validation

I run inference in sample-submission order. The script checks the exact header
and order, row count/IDs, finite probabilities, and the `[0, 1]` range before
reporting success. A missing selected series or invalid DICOM fails explicitly.
""",
        ),
        _cell(
            "code",
            """from scripts.predict_baseline import generate_submission

submission_path = WORK_ROOT / "submission.csv"
submission_report = generate_submission(
    DATA_ROOT,
    Path(final_result["checkpoint"]),
    submission_path,
    batch_size=1,
    device_name="auto",
    series_csv=test_manifest,
    series_directory_name=test_series_directory,
)
if submission_path.resolve() != Path("/kaggle/working/submission.csv").resolve():
    import shutil
    shutil.copyfile(submission_path, "/kaggle/working/submission.csv")
print(json.dumps(submission_report, indent=2))
assert submission_report["validation"]["valid"]
assert submission_report["validation"]["row_count"] == len(test_study_ids)
""",
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
            "kaggle": {
                "id": KERNEL_ID,
                "accelerator": "gpu",
                "internet": False,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    metadata = {
        "id": KERNEL_ID,
        "title": "KneeScope-12 MRI-only baseline",
        "code_file": NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "kernel_sources": [],
    }
    (NOTEBOOK.parent / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"I wrote {NOTEBOOK}")


if __name__ == "__main__":
    main()
