"""I provide synthetic DICOM fixtures without requiring competition data."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid


@pytest.fixture
def write_dicom():
    """I create a small metadata-only DICOM file for isolated infrastructure tests."""

    def _write(
        path: Path,
        *,
        study_uid: str | None = None,
        series_uid: str | None = None,
        instance_number: int = 1,
        position_z: float = 0.0,
    ) -> Path:
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
        dataset.SOPClassUID = SecondaryCaptureImageStorage
        dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        dataset.StudyInstanceUID = study_uid or generate_uid()
        dataset.SeriesInstanceUID = series_uid or generate_uid()
        dataset.Modality = "MR"
        dataset.SeriesDescription = "SAG PD"
        dataset.ProtocolName = "Knee protocol"
        dataset.SeriesNumber = 1
        dataset.InstanceNumber = instance_number
        dataset.Rows = 64
        dataset.Columns = 64
        dataset.PixelSpacing = [0.7, 0.7]
        dataset.SliceThickness = 3.0
        dataset.SpacingBetweenSlices = 3.0
        dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        dataset.ImagePositionPatient = [0, 0, position_z]
        dataset.Laterality = "R"
        dataset.AcquisitionDate = "20260101"
        dataset.AcquisitionTime = "120000"
        dataset.save_as(str(path), enforce_file_format=True)
        return path

    return _write
