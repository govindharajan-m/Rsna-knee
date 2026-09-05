"""I reconstruct validated MRI series into raw NumPy volumes without resizing them."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kneescope12.utils.logging import get_logger

from .dicom import DicomMetadata, read_dicom_image, read_dicom_metadata
from .ordering import SliceOrdering, order_slices

logger = get_logger("volume")


@dataclass(frozen=True)
class VolumeGeometry:
    """I preserve original DICOM geometry separately from later processed arrays."""

    study_instance_uid: str | None
    series_instance_uid: str | None
    series_description: str | None
    laterality: str | None
    original_dimensions: tuple[int, int] | None
    voxel_spacing_mm: tuple[float | None, float | None, float | None]
    orientation: tuple[float, float, float, float, float, float] | None
    slice_positions: tuple[tuple[float, float, float] | None, ...]


@dataclass(frozen=True)
class VolumeReconstruction:
    """I retain one raw volume, its source geometry, and any construction warnings."""

    raw_volume: np.ndarray | None
    geometry: VolumeGeometry | None
    ordering: SliceOrdering
    source_paths: tuple[Path, ...]
    skipped_paths: tuple[Path, ...]
    errors: tuple[str, ...]
    quality_flags: tuple[str, ...]

    @property
    def is_usable(self) -> bool:
        """I report whether reconstruction produced at least one consistent MRI slice."""

        return self.raw_volume is not None and self.geometry is not None


def reconstruct_series_from_paths(paths: Sequence[str | Path]) -> VolumeReconstruction:
    """I collect readable metadata before reconstructing a series from DICOM paths."""

    metadata: list[DicomMetadata] = []
    errors: list[str] = []
    skipped: list[Path] = []
    for path in paths:
        result = read_dicom_metadata(path)
        if result.metadata is None:
            skipped.append(Path(path))
            errors.append(f"{path}: {result.error or 'metadata read failed'}")
        else:
            metadata.append(result.metadata)
    reconstruction = reconstruct_series(metadata)
    return VolumeReconstruction(
        raw_volume=reconstruction.raw_volume,
        geometry=reconstruction.geometry,
        ordering=reconstruction.ordering,
        source_paths=reconstruction.source_paths,
        skipped_paths=tuple((*skipped, *reconstruction.skipped_paths)),
        errors=tuple((*errors, *reconstruction.errors)),
        quality_flags=reconstruction.quality_flags,
    )


def reconstruct_series(slices: Sequence[DicomMetadata]) -> VolumeReconstruction:
    """I spatially order and stack a valid, dimension-consistent MRI series.

    I do not force a common image shape. I skip decodable slices whose dimensions
    disagree with the first decoded slice and record that decision as a QC flag.
    """

    ordering = order_slices(slices)
    source_paths = tuple(item.path for item in ordering.ordered_slices)
    if not ordering.ordered_slices:
        return VolumeReconstruction(None, None, ordering, (), (), (), ordering.quality_flags)

    decoded_slices: list[np.ndarray] = []
    decoded_metadata: list[DicomMetadata] = []
    skipped_paths: list[Path] = []
    errors: list[str] = []
    flags = set(ordering.quality_flags)
    expected_shape: tuple[int, int] | None = None
    for item in ordering.ordered_slices:
        result = read_dicom_image(item.path)
        if result.pixels is None:
            skipped_paths.append(item.path)
            errors.append(f"{item.path}: {result.error or 'pixel decoding failed'}")
            flags.add("unreadable_slice_pixels")
            continue
        shape = tuple(result.pixels.shape)
        if expected_shape is None:
            expected_shape = shape
        if shape != expected_shape:
            skipped_paths.append(item.path)
            errors.append(f"{item.path}: dimensions {shape} differ from {expected_shape}")
            flags.add("inconsistent_decoded_dimensions")
            continue
        decoded_slices.append(result.pixels)
        decoded_metadata.append(item)
    if not decoded_slices:
        flags.add("no_decodable_slices")
        return VolumeReconstruction(
            None,
            None,
            ordering,
            source_paths,
            tuple(skipped_paths),
            tuple(errors),
            tuple(sorted(flags)),
        )

    raw_volume = np.stack(decoded_slices, axis=0)
    geometry = _build_geometry(decoded_metadata, ordering)
    if skipped_paths:
        logger.warning("I skipped %d slices while reconstructing series %s", len(skipped_paths), geometry.series_instance_uid)
    return VolumeReconstruction(
        raw_volume=raw_volume,
        geometry=geometry,
        ordering=ordering,
        source_paths=source_paths,
        skipped_paths=tuple(skipped_paths),
        errors=tuple(errors),
        quality_flags=tuple(sorted(flags)),
    )


def _build_geometry(slices: Sequence[DicomMetadata], ordering: SliceOrdering) -> VolumeGeometry:
    """I create geometry from the decoded slices and the selected ordering evidence."""

    first = slices[0]
    row_spacing = first.pixel_spacing[0] if first.pixel_spacing is not None else None
    column_spacing = first.pixel_spacing[1] if first.pixel_spacing is not None else None
    slice_spacing = ordering.inferred_slice_spacing_mm
    if slice_spacing is None:
        slice_spacing = first.spacing_between_slices or first.slice_thickness
    return VolumeGeometry(
        study_instance_uid=first.study_instance_uid,
        series_instance_uid=first.series_instance_uid,
        series_description=first.series_description,
        laterality=first.laterality,
        original_dimensions=first.dimensions,
        voxel_spacing_mm=(slice_spacing, row_spacing, column_spacing),
        orientation=first.image_orientation_patient,
        slice_positions=tuple(item.image_position_patient for item in slices),
    )
