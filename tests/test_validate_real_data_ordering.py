"""I test deterministic metadata-only validation of real-data series."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_validator():
    script = Path(__file__).parents[1] / "scripts" / "validate_real_data_ordering.py"
    specification = importlib.util.spec_from_file_location("validate_real_data_ordering", script)
    if specification is None or specification.loader is None:
        raise AssertionError(f"I could not load validator module from {script}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_validate_reports_adjacent_spacing_for_ordered_series(tmp_path, write_dicom):
    """I calculate adjacent spacing without requiring offset iterables to have equal lengths."""

    validator = _load_validator()
    dataset_root = tmp_path / "dataset"
    series_directory = dataset_root / "train_series" / "study-1" / "series-1"
    series_directory.mkdir(parents=True)
    for name, position in (("slice-1.dcm", 1.0), ("slice-2.dcm", 4.0), ("slice-3.dcm", 7.0)):
        write_dicom(series_directory / name, position_z=position)
    series_csv = dataset_root / "train_series.csv"
    series_csv.write_text(
        "StudyInstanceUID,SeriesInstanceUID,anatomical_plane,fluid_sensitive\n"
        "study-1,series-1,sagittal,yes\n",
        encoding="utf-8",
    )

    report = validator.validate(dataset_root, series_csv)

    assert report["geometry_failure_count"] == 0
    assert report["series_successfully_validated"] == 1
    assert report["adjacent_physical_spacing"] == {
        "minimum": 3.0,
        "median": 3.0,
        "maximum": 3.0,
    }
    assert report["physical_extent"] == {
        "minimum": 6.0,
        "median": 6.0,
        "maximum": 6.0,
    }
