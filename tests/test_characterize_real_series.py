"""I test bounded grouping and metadata-only real-series characterization."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from pydicom import dcmread


def _load_characterizer():
    script = Path(__file__).parents[1] / "scripts" / "characterize_real_series.py"
    specification = importlib.util.spec_from_file_location("characterize_real_series", script)
    if specification is None or specification.loader is None:
        raise AssertionError(f"I could not load characterizer module from {script}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_characterize_caps_at_60_series_with_deterministic_multi_series_sampling(tmp_path):
    """I sample multiple-series groups deterministically without exceeding 60 series."""

    characterizer = _load_characterizer()
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    series_csv = dataset_root / "train_series.csv"
    families = (
        ("Axial", "0", "0"),
        ("Axial", "1", "1"),
        ("Coronal", "1", "0"),
        ("Sagittal", "0", "1"),
    )
    rows = [
        f"study-{study_index:02d},series-{series_index},{fluid_sensitive},"
        f"{fat_suppression},{plane}"
        for study_index in range(21)
        for plane, fluid_sensitive, fat_suppression in families
        for series_index in range(4)
    ]
    series_csv.write_text(
        "StudyInstanceUID,SeriesInstanceUID,Fluid_Sensitive,Fat_Suppression,"
        "Anatomical_Plane\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )

    report = characterizer.characterize(dataset_root, series_csv)
    repeated_report = characterizer.characterize(dataset_root, series_csv)
    selected_ids = [
        (series["study_id"], series["series_id"])
        for series in report["series_characteristics"]
    ]
    expected_ids = [
        (f"study-{study_index:02d}", f"series-{series_index}")
        for study_index in range(5)
        for plane, fluid_sensitive, fat_suppression in sorted(families)
        for series_index in range(3)
    ]
    expected_group_keys = [
        (f"study-{study_index:02d}", plane, fluid_sensitive, fat_suppression)
        for study_index in range(5)
        for plane, fluid_sensitive, fat_suppression in sorted(families)
    ]

    assert report["sampling"]["manifest_group_count"] == 21 * len(families)
    assert report["sampling"]["selected_series_count"] == 60
    assert len(selected_ids) == 60
    assert selected_ids == expected_ids
    assert [
        (
            group["study_id"],
            group["anatomical_plane"],
            group["fluid_sensitive"],
            group["fat_suppression"],
        )
        for group in report["selected_groups"]
    ] == expected_group_keys
    assert len(set(expected_group_keys)) == len(expected_group_keys)
    assert selected_ids == [
        (series["study_id"], series["series_id"])
        for series in repeated_report["series_characteristics"]
    ]
    assert {series["group_size"] for series in report["series_characteristics"]} == {4}
    selected_per_group = {
        series["selected_series_from_group"] for series in report["series_characteristics"]
    }
    assert selected_per_group == {3}


def test_characterize_reports_geometry_metrics_duplicates_and_metadata_fallback(
    tmp_path, write_dicom
):
    """I report numeric metadata, duplicate coordinates, and failed-ordering metadata."""

    characterizer = _load_characterizer()
    dataset_root = tmp_path / "dataset"
    rows = []
    series_specs = {
        "series-duplicate": ((1.0, 3.0, 3.0), (0.8, 1.2), 4.0),
        "series-missing-orientation": ((5.0, 7.0), (0.6, 1.0), 2.5),
    }
    missing_orientation_path = None
    for series_id, (positions, spacing, thickness) in series_specs.items():
        series_directory = dataset_root / "train_series" / "study-1" / series_id
        series_directory.mkdir(parents=True)
        for index, position in enumerate(positions, start=1):
            path = write_dicom(
                series_directory / f"slice-{index}.dcm",
                position_z=position,
            )
            dataset = dcmread(path)
            dataset.PixelSpacing = list(spacing)
            dataset.SliceThickness = thickness
            if series_id == "series-missing-orientation" and index == 1:
                del dataset.ImageOrientationPatient
                missing_orientation_path = path
            dataset.save_as(path, enforce_file_format=True)
        rows.append(f"study-1,{series_id},1,0,Axial")
    assert missing_orientation_path is not None

    series_csv = dataset_root / "train_series.csv"
    series_csv.write_text(
        "StudyInstanceUID,SeriesInstanceUID,Fluid_Sensitive,Fat_Suppression,"
        "Anatomical_Plane\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )

    report = characterizer.characterize(dataset_root, series_csv)

    assert report["sampling"]["manifest_group_count"] == 1
    assert report["sampling"]["multi_series_group_count"] == 1
    assert report["sampling"]["selected_series_count"] == 2
    assert report["geometry_ordering_success_count"] == 1
    assert report["geometry_ordering_failure_count"] == 1
    duplicate, failed = report["series_characteristics"]
    assert duplicate["series_id"] == "series-duplicate"
    assert duplicate["geometry_ordering_succeeds"] is True
    assert duplicate["slice_count"] == 3
    assert duplicate["pixel_spacing_mm"] == {"row_mm": 0.8, "column_mm": 1.2}
    assert duplicate["slice_thickness_mm"] == 4.0
    assert duplicate["physical_adjacent_spacing_mm"] == [2.0, 0.0]
    assert duplicate["physical_extent_mm"] == 2.0
    assert duplicate["duplicate_physical_coordinates"] is True
    assert duplicate["geometry_metadata_presence"]["ImagePositionPatient"] == {
        "present": 3,
        "absent": 0,
    }
    assert failed["series_id"] == "series-missing-orientation"
    assert failed["geometry_ordering_succeeds"] is False
    assert "DicomGeometryError" in failed["geometry_ordering_error"]
    assert failed["metadata_unreadable_slices"] == 0
    assert failed["metadata_read_slices"] == 2
    assert failed["pixel_spacing_mm"] == {"row_mm": 0.6, "column_mm": 1.0}
    assert failed["slice_thickness_mm"] == 2.5
    assert failed["geometry_metadata_presence"]["ImageOrientationPatient"] == {
        "present": 1,
        "absent": 1,
    }
    assert report["geometry_metadata_presence"]["ImageOrientationPatient"] == {
        "present": 4,
        "absent": 1,
    }
    assert report["geometry_metadata_presence"]["PixelSpacing"] == {
        "present": 5,
        "absent": 0,
    }
    assert report["metric_distributions"]["slice_count"] == {
        "count": 2,
        "minimum": 2,
        "median": 2.5,
        "maximum": 3,
    }
    assert report["metric_distributions"]["pixel_spacing_row_mm"] == {
        "count": 2,
        "minimum": 0.6,
        "median": 0.7,
        "maximum": 0.8,
    }
    assert report["metric_distributions"]["pixel_spacing_column_mm"] == {
        "count": 2,
        "minimum": 1.0,
        "median": 1.1,
        "maximum": 1.2,
    }
    assert report["metric_distributions"]["slice_thickness_mm"] == {
        "count": 2,
        "minimum": 2.5,
        "median": 3.25,
        "maximum": 4.0,
    }
    assert report["metric_distributions"]["physical_adjacent_spacing_mm"] == {
        "count": 2,
        "minimum": 0.0,
        "median": 1.0,
        "maximum": 2.0,
    }
    assert report["metric_distributions"]["physical_extent_mm"] == {
        "count": 1,
        "minimum": 2.0,
        "median": 2.0,
        "maximum": 2.0,
    }
    assert report["image_dimensions_counts"] == {"64x64": 5}
    assert report["geometry_metadata_presence"]["ImagePositionPatient"] == {
        "present": 5,
        "absent": 0,
    }
