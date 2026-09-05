"""I test safe metadata extraction and spatial slice sorting."""

from __future__ import annotations

from kneescope12.data.dicom import read_dicom_metadata, sort_series_slices


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
