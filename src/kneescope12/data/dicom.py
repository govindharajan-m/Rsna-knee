"""I expose small, metadata-only DICOM interfaces for later MRI preprocessing."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydicom import dcmread
from pydicom.errors import InvalidDicomError

from kneescope12.utils.logging import get_logger

logger = get_logger("dicom")

_METADATA_TAGS = [
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "Modality",
    "SeriesDescription",
    "ProtocolName",
    "SeriesNumber",
    "InstanceNumber",
    "Rows",
    "Columns",
    "PixelSpacing",
    "SliceThickness",
    "SpacingBetweenSlices",
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "SliceLocation",
    "Laterality",
    "ImageLaterality",
    "AcquisitionDate",
    "AcquisitionTime",
    "Manufacturer",
    "ManufacturerModelName",
    "MagneticFieldStrength",
    "SequenceName",
]


@dataclass(frozen=True)
class DicomMetadata:
    """I retain the metadata that the initial audit needs from one DICOM file."""

    path: Path
    study_instance_uid: str | None
    series_instance_uid: str | None
    sop_instance_uid: str | None
    modality: str | None
    series_description: str | None
    protocol_name: str | None
    series_number: int | None
    instance_number: int | None
    rows: int | None
    columns: int | None
    pixel_spacing: tuple[float, float] | None
    slice_thickness: float | None
    spacing_between_slices: float | None
    image_position_patient: tuple[float, float, float] | None
    image_orientation_patient: tuple[float, float, float, float, float, float] | None
    slice_location: float | None
    laterality: str | None
    acquisition_date: str | None
    acquisition_time: str | None
    manufacturer: str | None
    manufacturer_model_name: str | None
    magnetic_field_strength: float | None
    sequence_name: str | None

    @property
    def dimensions(self) -> tuple[int, int] | None:
        """I return image rows and columns only when both are available."""

        if self.rows is None or self.columns is None:
            return None
        return self.rows, self.columns

    @property
    def is_image(self) -> bool:
        """I identify a likely image object from its image dimensions."""

        return self.rows is not None and self.columns is not None

    def to_dict(self) -> dict[str, Any]:
        """I return a JSON-serializable representation of the metadata."""

        result = asdict(self)
        result["path"] = str(self.path)
        return result


@dataclass(frozen=True)
class DicomReadResult:
    """I retain either metadata or an error instead of dropping an unreadable file."""

    path: Path
    metadata: DicomMetadata | None
    error: str | None

    @property
    def succeeded(self) -> bool:
        """I report whether metadata extraction succeeded."""

        return self.metadata is not None


class DicomGeometryError(ValueError):
    """I report missing or invalid geometry required for physical ordering."""


@dataclass(frozen=True)
class OrderedDicomSlice:
    """I retain one path, its metadata, and its physical slice coordinate."""

    path: Path
    metadata: DicomMetadata
    physical_coordinate: float
    duplicate_physical_coordinate: bool = False


@dataclass(frozen=True)
class DicomSliceOrdering:
    """I retain ordered slices, the normalized normal, and duplicate information."""

    slices: tuple[OrderedDicomSlice, ...]
    slice_normal: tuple[float, float, float]
    has_duplicate_physical_coordinates: bool


def order_dicom_slices(paths: Sequence[str | Path]) -> DicomSliceOrdering:
    """I order DICOM paths by their physical slice coordinates.

    I read metadata only; pixel arrays are never materialized. Duplicate physical
    coordinates remain in the result and use InstanceNumber, then filepath, as
    deterministic secondary keys. I flag every member of a duplicate group.
    """

    metadata: list[DicomMetadata] = []
    for path in paths:
        result = read_dicom_metadata(path)
        if result.metadata is None:
            raise DicomGeometryError(
                f"I could not read geometry from {Path(path)}: {result.error or 'metadata read failed'}"
            )
        metadata.append(result.metadata)
    if not metadata:
        raise DicomGeometryError("I cannot order an empty DICOM series")

    normal = _validated_slice_normal(metadata[0])
    coordinates: list[float] = []
    for item in metadata:
        item_normal = _validated_slice_normal(item)
        if not np.allclose(item_normal, normal, rtol=1e-5, atol=1e-7):
            raise DicomGeometryError(f"I found inconsistent slice orientation in {item.path}")
        if item.image_position_patient is None:
            raise DicomGeometryError(f"I found missing ImagePositionPatient in {item.path}")
        position = np.asarray(item.image_position_patient, dtype=float)
        if not np.all(np.isfinite(position)):
            raise DicomGeometryError(f"I found non-finite ImagePositionPatient in {item.path}")
        coordinates.append(float(np.dot(position, normal)))

    indexed = list(zip(metadata, coordinates, strict=True))
    indexed.sort(key=lambda entry: (entry[1], _instance_sort_key(entry[0]), str(entry[0].path)))
    coordinate_counts: dict[float, int] = {}
    for _, coordinate in indexed:
        coordinate_counts[coordinate] = coordinate_counts.get(coordinate, 0) + 1
    slices = tuple(
        OrderedDicomSlice(
            path=item.path,
            metadata=item,
            physical_coordinate=coordinate,
            duplicate_physical_coordinate=coordinate_counts[coordinate] > 1,
        )
        for item, coordinate in indexed
    )
    return DicomSliceOrdering(
        slices=slices,
        slice_normal=tuple(float(value) for value in normal),
        has_duplicate_physical_coordinates=any(
            count > 1 for count in coordinate_counts.values()
        ),
    )


@dataclass(frozen=True)
class DicomImageReadResult:
    """I retain decoded pixels with metadata or a recoverable decoding error."""

    path: Path
    metadata: DicomMetadata | None
    pixels: np.ndarray | None
    error: str | None

    @property
    def succeeded(self) -> bool:
        """I report whether one image object supplied two-dimensional pixels."""

        return self.metadata is not None and self.pixels is not None


@dataclass(frozen=True)
class SeriesQuality:
    """I retain conservative flags that make a series worth later inspection."""

    series_instance_uid: str | None
    flags: tuple[str, ...]


def read_dicom_metadata(path: str | Path) -> DicomReadResult:
    """I read DICOM metadata without materializing pixel arrays.

    I convert expected parsing and filesystem failures into an error-bearing
    result. I log the failure so dataset scans retain an audit trail.
    """

    file_path = Path(path)
    try:
        dataset = dcmread(
            file_path,
            stop_before_pixels=True,
            force=False,
            specific_tags=_METADATA_TAGS,
        )
        return DicomReadResult(
            path=file_path, metadata=_extract_metadata(file_path, dataset), error=None
        )
    except (InvalidDicomError, OSError, ValueError, TypeError) as error:
        message = f"{type(error).__name__}: {error}"
        logger.warning("I could not read DICOM metadata from %s: %s", file_path, message)
        return DicomReadResult(path=file_path, metadata=None, error=message)


def is_dicom_file(path: str | Path) -> bool:
    """I identify a readable DICOM file using the same safe reader as the audit."""

    return read_dicom_metadata(path).succeeded


def read_dicom_image(path: str | Path) -> DicomImageReadResult:
    """I decode one DICOM image only when volume reconstruction explicitly requests it."""

    metadata_result = read_dicom_metadata(path)
    if not metadata_result.succeeded or metadata_result.metadata is None:
        return DicomImageReadResult(
            path=Path(path), metadata=None, pixels=None, error=metadata_result.error
        )
    try:
        dataset = dcmread(path, force=False)
        pixels = np.asarray(dataset.pixel_array)
        pixels = np.squeeze(pixels)
        if pixels.ndim != 2:
            raise ValueError(f"I expected a two-dimensional MRI slice, received shape {pixels.shape}")
        return DicomImageReadResult(
            path=Path(path), metadata=metadata_result.metadata, pixels=pixels, error=None
        )
    except (InvalidDicomError, OSError, ValueError, TypeError, AttributeError) as error:
        message = f"{type(error).__name__}: {error}"
        logger.warning("I could not decode DICOM pixels from %s: %s", path, message)
        return DicomImageReadResult(
            path=Path(path), metadata=metadata_result.metadata, pixels=None, error=message
        )


def sort_series_slices(items: Iterable[DicomMetadata]) -> list[DicomMetadata]:
    """I sort slices by spatial projection before using weaker fallback keys."""

    return sorted(items, key=_slice_sort_key)


def identify_series_quality(items: Iterable[DicomMetadata]) -> SeriesQuality:
    """I flag missing and internally inconsistent spatial metadata in one series."""

    slices = list(items)
    flags: list[str] = []
    series_uids = {item.series_instance_uid for item in slices if item.series_instance_uid}
    if not series_uids:
        flags.append("missing_series_instance_uid")
    if len(series_uids) > 1:
        flags.append("mixed_series_instance_uid")
    dimensions = {item.dimensions for item in slices if item.dimensions is not None}
    if not dimensions:
        flags.append("missing_image_dimensions")
    elif len(dimensions) > 1:
        flags.append("inconsistent_image_dimensions")
    spacings = {item.pixel_spacing for item in slices if item.pixel_spacing is not None}
    if not spacings:
        flags.append("missing_pixel_spacing")
    elif len(spacings) > 1:
        flags.append("inconsistent_pixel_spacing")
    orientations = {
        item.image_orientation_patient
        for item in slices
        if item.image_orientation_patient is not None
    }
    if not orientations:
        flags.append("missing_orientation")
    elif len(orientations) > 1:
        flags.append("inconsistent_orientation")
    positions = [
        item.image_position_patient for item in slices if item.image_position_patient is not None
    ]
    if len(positions) != len(slices):
        flags.append("missing_slice_position")
    if len(set(positions)) != len(positions):
        flags.append("duplicate_slice_position")
    return SeriesQuality(
        series_instance_uid=next(iter(series_uids), None), flags=tuple(sorted(set(flags)))
    )


def _extract_metadata(path: Path, dataset: Any) -> DicomMetadata:
    """I normalize selected DICOM values into stable Python scalar types."""

    laterality = _text(dataset, "ImageLaterality") or _text(dataset, "Laterality")
    return DicomMetadata(
        path=path,
        study_instance_uid=_text(dataset, "StudyInstanceUID"),
        series_instance_uid=_text(dataset, "SeriesInstanceUID"),
        sop_instance_uid=_text(dataset, "SOPInstanceUID"),
        modality=_text(dataset, "Modality"),
        series_description=_text(dataset, "SeriesDescription"),
        protocol_name=_text(dataset, "ProtocolName"),
        series_number=_integer(dataset, "SeriesNumber"),
        instance_number=_integer(dataset, "InstanceNumber"),
        rows=_integer(dataset, "Rows"),
        columns=_integer(dataset, "Columns"),
        pixel_spacing=_float_tuple(dataset, "PixelSpacing", length=2),
        slice_thickness=_float(dataset, "SliceThickness"),
        spacing_between_slices=_float(dataset, "SpacingBetweenSlices"),
        image_position_patient=_float_tuple(dataset, "ImagePositionPatient", length=3),
        image_orientation_patient=_float_tuple(dataset, "ImageOrientationPatient", length=6),
        slice_location=_float(dataset, "SliceLocation"),
        laterality=laterality,
        acquisition_date=_text(dataset, "AcquisitionDate"),
        acquisition_time=_text(dataset, "AcquisitionTime"),
        manufacturer=_text(dataset, "Manufacturer"),
        manufacturer_model_name=_text(dataset, "ManufacturerModelName"),
        magnetic_field_strength=_float(dataset, "MagneticFieldStrength"),
        sequence_name=_text(dataset, "SequenceName"),
    )


def _validated_slice_normal(item: DicomMetadata) -> np.ndarray:
    """I validate orientation vectors and return a normalized slice normal."""

    orientation = item.image_orientation_patient
    if orientation is None or len(orientation) != 6:
        raise DicomGeometryError(f"I found missing or invalid ImageOrientationPatient in {item.path}")
    row = np.asarray(orientation[:3], dtype=float)
    column = np.asarray(orientation[3:], dtype=float)
    if not np.all(np.isfinite(row)) or not np.all(np.isfinite(column)):
        raise DicomGeometryError(f"I found non-finite ImageOrientationPatient in {item.path}")
    if np.linalg.norm(row) <= 1e-8 or np.linalg.norm(column) <= 1e-8:
        raise DicomGeometryError(f"I found a zero orientation vector in {item.path}")
    normal = np.cross(row, column)
    normal_norm = np.linalg.norm(normal)
    if normal_norm <= 1e-8:
        raise DicomGeometryError(f"I found unusable orientation vectors in {item.path}")
    return normal / normal_norm


def _instance_sort_key(item: DicomMetadata) -> int:
    """I provide a stable secondary key without making it authoritative."""

    return item.instance_number if item.instance_number is not None else 0


def _slice_sort_key(item: DicomMetadata) -> tuple[int, float, int, str]:
    """I prefer the normal-axis projection defined by image orientation and position."""

    if item.image_position_patient is not None and item.image_orientation_patient is not None:
        orientation = np.asarray(item.image_orientation_patient, dtype=float)
        normal = np.cross(orientation[:3], orientation[3:])
        if not np.allclose(normal, 0.0):
            return (
                0,
                float(np.dot(np.asarray(item.image_position_patient), normal)),
                0,
                str(item.path),
            )
    if item.slice_location is not None:
        return (1, item.slice_location, item.instance_number or 0, str(item.path))
    if item.instance_number is not None:
        return (2, float(item.instance_number), item.instance_number, str(item.path))
    return (3, 0.0, 0, str(item.path))


def _text(dataset: Any, name: str) -> str | None:
    value = getattr(dataset, name, None)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _integer(dataset: Any, name: str) -> int | None:
    value = getattr(dataset, name, None)
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _float(dataset: Any, name: str) -> float | None:
    value = getattr(dataset, name, None)
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _float_tuple(dataset: Any, name: str, *, length: int) -> tuple[float, ...] | None:
    value = getattr(dataset, name, None)
    if value is None:
        return None
    try:
        converted = tuple(float(item) for item in value)
    except (TypeError, ValueError, OverflowError):
        return None
    return converted if len(converted) == length else None
