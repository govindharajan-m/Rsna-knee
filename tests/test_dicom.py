"""I test safe metadata extraction and spatial slice sorting."""

from __future__ import annotations

import pytest
from pydicom import dcmread

from kneescope12.data.dicom import (
    DicomGeometryError,
    order_dicom_slices,
    read_dicom_metadata,
    sort_series_slices,
)


def test_read_dicom_metadata_extracts_spatial_fields(tmp_path, write_dicom):
    """I extract dimensions, spacing, and position without loading pixel data."""

    result = read_dicom_metadata(write_dicom(tmp_path / "image.dcm", position_z=4.0))
    assert result.succeeded
    assert result.metadata is not None
    assert result.metadata.dimensions == (64, 64)
    assert result.metadata.pixel_spacing == (0.7, 0.7)
    assert result.metadata.image_position_patient == (0.0, 0.0, 4.0)


def test_sort_series_slices_prefers_spatial_position(tmp_path, write_dicom):
    """I sort slices by geometry rather than the incidental filename order."""

    high = read_dicom_metadata(write_dicom(tmp_path / "high.dcm", position_z=5.0)).metadata
    low = read_dicom_metadata(write_dicom(tmp_path / "low.dcm", position_z=1.0)).metadata
    assert high is not None and low is not None
    assert sort_series_slices([high, low]) == [low, high]


def test_malformed_dicom_returns_an_error(tmp_path):
    """I retain a malformed DICOM failure instead of raising or ignoring it."""

    malformed = tmp_path / "broken.dcm"
    malformed.write_text("not a DICOM file", encoding="utf-8")
    result = read_dicom_metadata(malformed)
    assert not result.succeeded
    assert result.metadata is None
    assert result.error is not None


def test_order_dicom_slices_uses_physical_geometry_not_filenames(tmp_path, write_dicom):
    """I order deliberately scrambled filenames by physical position."""

    paths = [
        write_dicom(tmp_path / "slice-10.dcm", position_z=10.0, instance_number=1),
        write_dicom(tmp_path / "slice-02.dcm", position_z=2.0, instance_number=2),
        write_dicom(tmp_path / "slice-07.dcm", position_z=7.0, instance_number=3),
    ]
    ordering = order_dicom_slices(paths)
    assert [item.physical_coordinate for item in ordering.slices] == [2.0, 7.0, 10.0]
    assert [item.path.name for item in ordering.slices] == [
        "slice-02.dcm",
        "slice-07.dcm",
        "slice-10.dcm",
    ]


def test_order_dicom_slices_ignores_conflicting_instance_numbers(tmp_path, write_dicom):
    """I keep physical coordinates authoritative when InstanceNumber conflicts."""

    paths = [
        write_dicom(tmp_path / "a.dcm", position_z=1.0, instance_number=20),
        write_dicom(tmp_path / "b.dcm", position_z=3.0, instance_number=10),
    ]
    ordering = order_dicom_slices(paths)
    assert [item.physical_coordinate for item in ordering.slices] == [1.0, 3.0]


def test_order_dicom_slices_reverses_stored_physical_order(tmp_path, write_dicom):
    """I return ascending physical coordinates for reverse-stored input."""

    paths = [
        write_dicom(tmp_path / "first.dcm", position_z=9.0),
        write_dicom(tmp_path / "second.dcm", position_z=6.0),
        write_dicom(tmp_path / "third.dcm", position_z=3.0),
    ]
    ordering = order_dicom_slices(paths)
    coordinates = [item.physical_coordinate for item in ordering.slices]
    assert coordinates == sorted(coordinates)
    assert coordinates == [3.0, 6.0, 9.0]


def test_order_dicom_slices_requires_image_position(tmp_path, write_dicom):
    """I reject a slice without ImagePositionPatient instead of guessing."""

    path = write_dicom(tmp_path / "missing-position.dcm")
    dataset = dcmread(path)
    del dataset.ImagePositionPatient
    dataset.save_as(path, enforce_file_format=True)
    with pytest.raises(DicomGeometryError, match="ImagePositionPatient"):
        order_dicom_slices([path])


def test_order_dicom_slices_requires_valid_orientation(tmp_path, write_dicom):
    """I reject invalid orientation vectors instead of falling back."""

    path = write_dicom(tmp_path / "invalid-orientation.dcm")
    dataset = dcmread(path)
    dataset.ImageOrientationPatient = [1, 0, 0, 0, 0, 0]
    dataset.save_as(path, enforce_file_format=True)
    with pytest.raises(DicomGeometryError, match="orientation"):
        order_dicom_slices([path])


def test_order_dicom_slices_retains_and_flags_duplicate_coordinates(tmp_path, write_dicom):
    """I retain duplicate physical coordinates and flag each duplicate slice."""

    paths = [
        write_dicom(tmp_path / "z.dcm", position_z=4.0, instance_number=2),
        write_dicom(tmp_path / "a.dcm", position_z=4.0, instance_number=1),
        write_dicom(tmp_path / "middle.dcm", position_z=2.0, instance_number=3),
    ]
    ordering = order_dicom_slices(paths)
    assert len(ordering.slices) == 3
    assert ordering.has_duplicate_physical_coordinates
    assert [item.physical_coordinate for item in ordering.slices] == [2.0, 4.0, 4.0]
    assert [item.path.name for item in ordering.slices] == ["middle.dcm", "a.dcm", "z.dcm"]
    assert [item.duplicate_physical_coordinate for item in ordering.slices] == [False, True, True]
