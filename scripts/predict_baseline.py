"""I generate and validate a continuous-probability competition submission."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from kneescope12.data.dataset import BaselineInferenceDataset  # noqa: E402
from kneescope12.model import BaselineClassifier  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--series-csv", type=Path)
    parser.add_argument("--series-directory-name", default="test_series")
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/submission.csv"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def _read_sample_submission(path: Path) -> tuple[list[str], list[str]]:
    """I retain the official column order and expected study row order."""

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            columns = next(reader)
        except StopIteration as error:
            raise ValueError(f"I found an empty sample submission: {path}") from error
        if len(columns) != 13 or columns[0] != "StudyInstanceUID":
            raise ValueError(
                "I expected sample_submission.csv to contain StudyInstanceUID "
                "and twelve prediction columns in the official order"
            )
        study_ids = [row[0].strip() for row in reader if row and row[0].strip()]
    if not study_ids:
        raise ValueError(f"I found no test study IDs in {path}")
    if len(set(study_ids)) != len(study_ids):
        raise ValueError(f"I found duplicate study IDs in {path}")
    return columns, study_ids


def validate_submission(
    sample_path: Path, submission_path: Path
) -> dict[str, int | bool]:
    """I verify exact columns, rows, IDs, finite probabilities, and probability range."""

    expected_columns, expected_ids = _read_sample_submission(sample_path)
    with submission_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_columns:
            raise ValueError(
                f"I found submission columns {reader.fieldnames}, expected {expected_columns}"
            )
        rows = list(reader)
    observed_ids = [row["StudyInstanceUID"] for row in rows]
    if observed_ids != expected_ids:
        raise ValueError(
            "I found submission study IDs/order that differ from sample_submission.csv"
        )
    for row_index, row in enumerate(rows, start=2):
        for column in expected_columns[1:]:
            try:
                value = float(row[column])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"I found a non-numeric prediction at row {row_index}, column {column}"
                ) from error
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"I found an invalid probability at row {row_index}, column {column}: {value}"
                )
    return {
        "valid": True,
        "row_count": len(rows),
        "prediction_column_count": len(expected_columns) - 1,
        "study_id_order_matches": True,
    }


def generate_submission(
    dataset_root: Path,
    checkpoint_path: Path,
    output_path: Path,
    *,
    batch_size: int = 1,
    device_name: str = "auto",
    series_csv: Path | None = None,
    series_directory_name: str = "test_series",
) -> dict[str, Any]:
    """I predict every sample-submission study and validate the completed CSV."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    sample_path = dataset_root / "sample_submission.csv"
    series_path = series_csv or (
        dataset_root / "test_series.csv"
        if (dataset_root / "test_series.csv").is_file()
        else None
    )
    if not sample_path.is_file():
        raise FileNotFoundError(f"I could not find official sample submission: {sample_path}")
    if series_path is not None and not series_path.is_file():
        raise FileNotFoundError(f"I could not find test series manifest: {series_path}")
    columns, study_ids = _read_sample_submission(sample_path)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("I found no baseline config in the model checkpoint")
    baseline = config["baseline"]
    preprocessing = baseline["preprocessing"]
    if list(config["labels"]) != columns[1:]:
        raise ValueError(
            "Checkpoint label order does not match the official sample submission columns"
        )
    dataset = BaselineInferenceDataset(
        dataset_root,
        series_path,
        study_ids,
        target_shape=tuple(int(value) for value in baseline["target_shape"]),
        plane_order=list(baseline["plane_order"]),
        max_series_per_study=int(baseline["max_series_per_study"]),
        series_directory_name=series_directory_name,
        plane_axis_mapping=checkpoint.get("plane_axis_mapping"),
        lower_percentile=float(preprocessing["lower_percentile"]),
        upper_percentile=float(preprocessing["upper_percentile"]),
        interpolation_order=int(preprocessing["interpolation_order"]),
    )
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("I was asked to use CUDA, but CUDA is unavailable")
    model = BaselineClassifier(
        in_channels=int(baseline["max_series_per_study"]),
        base_channels=int(config["model"]["base_channels"]),
        output_count=len(columns) - 1,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            suffix=".csv",
            dir=output_path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.writer(handle)
            writer.writerow(columns)
            with torch.no_grad():
                for batch_ids, images in loader:
                    probabilities = torch.sigmoid(model(images.to(device))).cpu().numpy()
                    if probabilities.shape != (len(batch_ids), len(columns) - 1):
                        raise RuntimeError(
                            f"I got unexpected prediction shape {probabilities.shape}"
                        )
                    if not np.all(np.isfinite(probabilities)):
                        raise ValueError("I generated non-finite predictions")
                    writer.writerows(
                        [study_id, *map(float, row)]
                        for study_id, row in zip(batch_ids, probabilities, strict=True)
                    )
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    validation = validate_submission(sample_path, output_path)
    return {
        "output": str(output_path),
        "checkpoint": str(checkpoint_path),
        "study_count": len(study_ids),
        "columns": columns,
        "validation": validation,
    }


def main() -> int:
    arguments = _parse_args()
    report = generate_submission(
        arguments.dataset_root.expanduser().resolve(),
        arguments.checkpoint.expanduser().resolve(),
        arguments.output.expanduser(),
        batch_size=arguments.batch_size,
        device_name=arguments.device,
        series_csv=arguments.series_csv.expanduser().resolve()
        if arguments.series_csv
        else None,
        series_directory_name=arguments.series_directory_name,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
