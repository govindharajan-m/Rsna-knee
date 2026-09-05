"""I order DICOM slices from spatial metadata and retain ordering evidence."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .dicom import DicomMetadata


@dataclass(frozen=True)
class SliceOrdering:
    """I describe the ordering method and geometry checks for one series."""

    ordered_slices: tuple[DicomMetadata, ...]
    strategy: str
    projected_positions: tuple[float | None, ...]
    inferred_slice_spacing_mm: float | None
    quality_flags: tuple[str, ...]


def order_slices(
    slices: Sequence[DicomMetadata], *, spacing_relative_tolerance: float = 0.15
) -> SliceOrdering:
    """I sort slices spatially whenever their orientations and positions allow it.

    I only fall back to SliceLocation, InstanceNumber, and finally the path when
    the preferred geometric fields are incomplete. The recorded strategy makes
    every reconstruction auditable.
    """

    if not slices:
        return SliceOrdering((), "empty", (), None, ("empty_series",))
    if spacing_relative_tolerance < 0:
        raise ValueError("spacing_relative_tolerance must be non-negative")

    spatial = _spatial_order(slices, spacing_relative_tolerance)
    if spatial is not None:
        return spatial
    if all(item.slice_location is not None for item in slices):
        ordered = tuple(sorted(slices, key=lambda item: (item.slice_location or 0.0, str(item.path))))
        positions = tuple(item.slice_location for item in ordered)
        return _finish_ordering(ordered, "slice_location", positions, spacing_relative_tolerance)
    if all(item.instance_number is not None for item in slices):
        ordered = tuple(sorted(slices, key=lambda item: (item.instance_number or 0, str(item.path))))
        positions = tuple(float(item.instance_number or 0) for item in ordered)
        result = _finish_ordering(ordered, "instance_number", positions, spacing_relative_tolerance)
        return _with_flag(result, "spatial_metadata_incomplete")

    ordered = tuple(sorted(slices, key=lambda item: str(item.path)))
    return SliceOrdering(
        ordered_slices=ordered,
        strategy="path_lexicographic",
        projected_positions=tuple(None for _ in ordered),
        inferred_slice_spacing_mm=None,
        quality_flags=("spatial_metadata_incomplete", "ordering_fallback_path"),
    )


def _spatial_order(
    slices: Sequence[DicomMetadata],
    spacing_relative_tolerance: float,
) -> SliceOrdering | None:
    """I derive a normal vector and project each ImagePositionPatient onto it."""

    if not all(
        item.image_position_patient is not None and item.image_orientation_patient is not None
        for item in slices
    ):
        return None
    first_orientation = np.asarray(slices[0].image_orientation_patient, dtype=float)
    normal = np.cross(first_orientation[:3], first_orientation[3:])
    if np.allclose(normal, 0.0):
        return None
    normal = normal / np.linalg.norm(normal)
    projections = [
        float(np.dot(np.asarray(item.image_position_patient, dtype=float), normal)) for item in slices
    ]
    order = np.argsort(projections, kind="stable")
    ordered = tuple(slices[index] for index in order)
    sorted_projections = tuple(projections[index] for index in order)
    result = _finish_ordering(
        ordered,
        "image_position_patient_projection",
        sorted_projections,
        spacing_relative_tolerance,
    )
    orientations = {item.image_orientation_patient for item in slices}
    if len(orientations) > 1:
        result = _with_flag(result, "inconsistent_orientation")
    if all(item.instance_number is not None for item in slices):
        spatial_instance_order = [item.instance_number for item in ordered]
        if spatial_instance_order != sorted(spatial_instance_order):
            result = _with_flag(result, "instance_number_disagrees_with_spatial_order")
    return result


def _finish_ordering(
    ordered: tuple[DicomMetadata, ...],
    strategy: str,
    positions: tuple[float | None, ...],
    spacing_relative_tolerance: float,
) -> SliceOrdering:
    """I calculate spacing consistency from the chosen scalar slice positions."""

    numeric_positions = [position for position in positions if position is not None]
    flags: list[str] = []
    inferred_spacing: float | None = None
    if len(numeric_positions) > 1:
        deltas = np.diff(np.asarray(numeric_positions, dtype=float))
        nonzero = np.abs(deltas[np.abs(deltas) > 1e-6])
        if len(nonzero) != len(deltas):
            flags.append("duplicate_slice_position")
        if len(nonzero):
            inferred_spacing = float(np.median(nonzero))
            allowed_deviation = max(inferred_spacing * spacing_relative_tolerance, 1e-6)
            if np.any(np.abs(nonzero - inferred_spacing) > allowed_deviation):
                flags.append("irregular_slice_spacing")
    elif len(ordered) > 1:
        flags.append("slice_spacing_unavailable")
    return SliceOrdering(
        ordered_slices=ordered,
        strategy=strategy,
        projected_positions=positions,
        inferred_slice_spacing_mm=inferred_spacing,
        quality_flags=tuple(sorted(set(flags))),
    )


def _with_flag(ordering: SliceOrdering, flag: str) -> SliceOrdering:
    """I add a deduplicated quality flag while preserving an immutable result."""

    return SliceOrdering(
        ordered_slices=ordering.ordered_slices,
        strategy=ordering.strategy,
        projected_positions=ordering.projected_positions,
        inferred_slice_spacing_mm=ordering.inferred_slice_spacing_mm,
        quality_flags=tuple(sorted(set((*ordering.quality_flags, flag)))),
    )
