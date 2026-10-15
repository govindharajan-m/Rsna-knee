"""I load explicitly labeled studies into deterministic MRI tensor examples."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .dicom import read_dicom_metadata
from .preprocessing import preprocess_baseline_volume
from .volume import reconstruct_ordered_volume


class BaselineStudyDataset(Dataset):
    """I yield tensors shaped ``[series channel, depth, row, column]`` and 12 labels.

    I select the lexicographically first SeriesInstanceUID for each manifest
    plane. This reproducible policy is not claimed to be optimal. Missing plane
    channels stay zero; invalid selected series raise instead of falling back.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        labels_csv: str | Path,
        series_csv: str | Path,
        study_ids: list[str],
        label_columns: list[str],
        *,
        target_shape: tuple[int, int, int],
        plane_order: list[str] | None = None,
        max_series_per_study: int = 3,
        series_directory_name: str = "train_series",
        lower_percentile: float = 0.5,
        upper_percentile: float = 99.5,
        interpolation_order: int = 1,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.target_shape = target_shape
        self.plane_order = plane_order or ["Axial", "Coronal", "Sagittal"]
        self.max_series_per_study = max_series_per_study
        self.series_directory_name = series_directory_name
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.interpolation_order = interpolation_order
        self.study_ids = list(study_ids)
        if max_series_per_study <= 0:
            raise ValueError("max_series_per_study must be positive")
        if len(label_columns) != 12 or len(set(label_columns)) != 12:
            raise ValueError("I require exactly 12 unique label columns")
        labels = read_gold_labels(Path(labels_csv), label_columns)
        absent_labels = [study_id for study_id in self.study_ids if study_id not in labels]
        if absent_labels:
            raise ValueError(
                "I found requested studies without explicit binary labels: "
                f"{absent_labels}"
            )
        self.targets = {study_id: labels[study_id] for study_id in self.study_ids}
        self.candidates = _read_series_candidates(Path(series_csv), self.study_ids)
        self._cache: dict[str, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.study_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        study_id = self.study_ids[index]
        if study_id not in self._cache:
            self._cache[study_id] = self._load_study(study_id)
        return self._cache[study_id], torch.from_numpy(self.targets[study_id].copy())

    def _load_study(self, study_id: str) -> torch.Tensor:
        return load_study_tensor(
            self.dataset_root,
            study_id,
            self.candidates.get(study_id, []),
            target_shape=self.target_shape,
            plane_order=self.plane_order,
            max_series_per_study=self.max_series_per_study,
            lower_percentile=self.lower_percentile,
            upper_percentile=self.upper_percentile,
            interpolation_order=self.interpolation_order,
            series_directory_name=self.series_directory_name,
        )


class BaselineInferenceDataset(Dataset):
    """I return study IDs and tensors without requiring or reading labels."""

    def __init__(
        self,
        dataset_root: str | Path,
        series_csv: str | Path | None,
        study_ids: list[str],
        *,
        target_shape: tuple[int, int, int],
        plane_order: list[str] | None = None,
        max_series_per_study: int = 3,
        series_directory_name: str = "test_series",
        plane_axis_mapping: dict[str, int] | None = None,
        lower_percentile: float = 0.5,
        upper_percentile: float = 99.5,
        interpolation_order: int = 1,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.study_ids = list(study_ids)
        self.target_shape = target_shape
        self.plane_order = plane_order or ["Axial", "Coronal", "Sagittal"]
        self.max_series_per_study = max_series_per_study
        self.series_directory_name = series_directory_name
        self.plane_axis_mapping = plane_axis_mapping
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.interpolation_order = interpolation_order
        self.candidates = (
            _read_series_candidates(Path(series_csv), self.study_ids)
            if series_csv is not None
            else None
        )

    def __len__(self) -> int:
        return len(self.study_ids)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor]:
        study_id = self.study_ids[index]
        candidates = (
            self.candidates.get(study_id, [])
            if self.candidates is not None
            else _discover_series_candidates(
                self.dataset_root,
                study_id,
                self.series_directory_name,
                self.plane_axis_mapping,
            )
        )
        tensor = load_study_tensor(
            self.dataset_root,
            study_id,
            candidates,
            target_shape=self.target_shape,
            plane_order=self.plane_order,
            max_series_per_study=self.max_series_per_study,
            lower_percentile=self.lower_percentile,
            upper_percentile=self.upper_percentile,
            interpolation_order=self.interpolation_order,
            series_directory_name=self.series_directory_name,
        )
        return study_id, tensor


def load_study_tensor(
    dataset_root: str | Path,
    study_id: str,
    candidates: list[dict[str, str]],
    *,
    target_shape: tuple[int, int, int],
    plane_order: list[str],
    max_series_per_study: int,
    series_directory_name: str,
    lower_percentile: float,
    upper_percentile: float,
    interpolation_order: int,
) -> torch.Tensor:
    """I apply fixed plane slots, strict reconstruction, and configured preprocessing."""

    by_plane: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        by_plane[row["Anatomical_Plane"]].append(row)
    unexpected_planes = sorted(set(by_plane) - set(plane_order))
    if unexpected_planes:
        raise ValueError(
            f"I found manifest planes missing from configured plane_order: {unexpected_planes}"
        )
    selected = [
        sorted(
            by_plane[plane],
            key=lambda row: row["SeriesInstanceUID"],
        )[0]
        if by_plane.get(plane)
        else None
        for plane in plane_order[:max_series_per_study]
    ]
    if all(row is None for row in selected):
        raise ValueError(f"I found no manifest-listed MRI series for study {study_id}")

    image = np.zeros(
        (max_series_per_study, *target_shape),
        dtype=np.float32,
    )
    for channel, row in enumerate(selected):
        if row is None:
            continue
        directory = (
            Path(dataset_root)
            / series_directory_name
            / study_id
            / row["SeriesInstanceUID"]
        )
        if not directory.is_dir():
            raise FileNotFoundError(f"I could not find selected series directory: {directory}")
        paths = sorted(path for path in directory.iterdir() if path.is_file())
        if not paths:
            raise ValueError(f"I found no DICOM files in selected series: {directory}")
        volume = reconstruct_ordered_volume(paths)
        image[channel] = preprocess_baseline_volume(
            volume,
            target_shape=target_shape,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
            interpolation_order=interpolation_order,
        )
    return torch.from_numpy(image)


def derive_plane_axis_mapping(
    dataset_root: str | Path,
    series_csv: str | Path,
    study_ids: list[str],
    plane_order: list[str],
    *,
    series_directory_name: str = "train_series",
) -> tuple[dict[str, int], dict[str, dict[int, int]]]:
    """I calibrate manifest plane names to dominant DICOM normal-coordinate axes.

    I use only each selected training series' first readable metadata file. The
    mapping is an empirical association from the manifest, not a patient-side
    interpretation. Every configured plane must map to a distinct x/y/z axis.
    """

    candidates = _read_series_candidates(Path(series_csv), study_ids)
    counts: dict[str, dict[int, int]] = {plane: defaultdict(int) for plane in plane_order}
    root = Path(dataset_root)
    for study_id in study_ids:
        by_plane: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in candidates.get(study_id, []):
            if row["Anatomical_Plane"] in counts:
                by_plane[row["Anatomical_Plane"]].append(row)
        for plane in plane_order:
            if not by_plane[plane]:
                continue
            selected = min(by_plane[plane], key=lambda row: row["SeriesInstanceUID"])
            directory = (
                root / series_directory_name / study_id / selected["SeriesInstanceUID"]
            )
            if not directory.is_dir():
                raise FileNotFoundError(
                    f"I could not find selected training series for plane-axis mapping: "
                    f"{directory}"
                )
            paths = sorted(path for path in directory.iterdir() if path.is_file())
            if not paths:
                raise ValueError(
                    f"I found no DICOM metadata files for plane-axis mapping: {directory}"
                )
            result = read_dicom_metadata(paths[0])
            if result.metadata is None or result.metadata.image_orientation_patient is None:
                raise ValueError(
                    f"I could not read ImageOrientationPatient for plane-axis mapping "
                    f"from {paths[0]}: {result.error or 'metadata is missing'}"
                )
            orientation = np.asarray(result.metadata.image_orientation_patient, dtype=float)
            if orientation.shape != (6,) or not np.all(np.isfinite(orientation)):
                raise ValueError(
                    f"I found invalid ImageOrientationPatient for plane-axis mapping "
                    f"in {paths[0]}"
                )
            normal = np.cross(orientation[:3], orientation[3:])
            norm = float(np.linalg.norm(normal))
            if norm <= 1e-8:
                raise ValueError(
                    f"I found degenerate ImageOrientationPatient for plane-axis mapping "
                    f"in {paths[0]}"
                )
            axis = int(np.argmax(np.abs(normal / norm)))
            counts[plane][axis] += 1
    mapping: dict[str, int] = {}
    for plane in plane_order:
        observed = counts[plane]
        if not observed:
            raise ValueError(f"I could not derive a DICOM normal axis for manifest plane {plane}")
        highest = max(observed.values())
        winners = [axis for axis, count in observed.items() if count == highest]
        if len(winners) != 1:
            raise ValueError(
                f"I found an ambiguous DICOM normal axis for {plane}: {dict(observed)}"
            )
        mapping[plane] = winners[0]
    if len(set(mapping.values())) != len(mapping):
        raise ValueError(f"I found overlapping plane/normal-axis mapping: {mapping}")
    return mapping, {plane: dict(sorted(value.items())) for plane, value in counts.items()}


def _discover_series_candidates(
    dataset_root: Path,
    study_id: str,
    series_directory_name: str,
    plane_axis_mapping: dict[str, int] | None,
) -> list[dict[str, str]]:
    """I classify test-series folders using the manifest-calibrated normal axis."""

    if not plane_axis_mapping:
        raise ValueError(
            "I need test-series manifest rows or a training-derived plane_axis_mapping"
        )
    axis_to_plane = {axis: plane for plane, axis in plane_axis_mapping.items()}
    study_directory = dataset_root / series_directory_name / study_id
    if not study_directory.is_dir():
        raise FileNotFoundError(f"I could not find test study directory: {study_directory}")
    candidates: list[dict[str, str]] = []
    for directory in sorted(path for path in study_directory.iterdir() if path.is_dir()):
        files = sorted(path for path in directory.iterdir() if path.is_file())
        if not files:
            raise ValueError(f"I found an empty test series directory: {directory}")
        result = read_dicom_metadata(files[0])
        if result.metadata is None or result.metadata.image_orientation_patient is None:
            raise ValueError(
                f"I could not derive orientation for test series {directory}: "
                f"{result.error or 'missing ImageOrientationPatient'}"
            )
        orientation = np.asarray(result.metadata.image_orientation_patient, dtype=float)
        if orientation.shape != (6,) or not np.all(np.isfinite(orientation)):
            raise ValueError(f"I found invalid ImageOrientationPatient in {files[0]}")
        normal = np.cross(orientation[:3], orientation[3:])
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-8:
            raise ValueError(f"I found degenerate ImageOrientationPatient in {files[0]}")
        axis = int(np.argmax(np.abs(normal / norm)))
        plane = axis_to_plane.get(axis)
        if plane is not None:
            candidates.append(
                {
                    "SeriesInstanceUID": directory.name,
                    "Fluid_Sensitive": "",
                    "Fat_Suppression": "",
                    "Anatomical_Plane": plane,
                }
            )
    if not candidates:
        raise ValueError(
            f"I could not assign any test series to configured planes for study {study_id}"
        )
    return candidates


def read_gold_labels(path: Path, label_columns: list[str]) -> dict[str, np.ndarray]:
    """I retain only studies having twelve explicit finite binary labels."""

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"I found no header in label table {path}")
        normalized = {name.strip().casefold(): name for name in reader.fieldnames}
        identifier = normalized.get("studyinstanceuid")
        if identifier is None:
            raise ValueError(f"I could not find StudyInstanceUID in {path}")
        missing = [name for name in label_columns if name.casefold() not in normalized]
        if missing:
            raise ValueError(f"I could not find configured label columns in {path}: {missing}")
        labels: dict[str, np.ndarray] = {}
        for row in reader:
            study_id = (row.get(identifier) or "").strip()
            if not study_id:
                continue
            raw_values = [
                (row.get(normalized[name.casefold()]) or "").strip()
                for name in label_columns
            ]
            if any(value == "" for value in raw_values):
                continue
            try:
                values = np.asarray([float(value) for value in raw_values], dtype=np.float32)
            except ValueError:
                continue
            if not np.all(np.isfinite(values)) or not np.all(np.isin(values, (0.0, 1.0))):
                continue
            if study_id in labels:
                raise ValueError(f"I found duplicate explicitly labeled study ID: {study_id}")
            labels[study_id] = values
    if not labels:
        raise ValueError(f"I found no studies with 12 explicit binary labels in {path}")
    return labels


def _read_series_candidates(
    path: Path, study_ids: list[str]
) -> dict[str, list[dict[str, str]]]:
    """I index manifest rows for requested studies without scanning DICOM folders."""

    requested = set(study_ids)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"I found no header in series manifest {path}")
        normalized = {name.strip().casefold(): name for name in reader.fieldnames}
        required = (
            "StudyInstanceUID",
            "SeriesInstanceUID",
            "Fluid_Sensitive",
            "Fat_Suppression",
            "Anatomical_Plane",
        )
        missing = [name for name in required if name.casefold() not in normalized]
        if missing:
            raise ValueError(f"I could not find required RSNA series columns: {missing}")
        candidates: dict[str, list[dict[str, str]]] = defaultdict(list)
        seen: set[tuple[str, str]] = set()
        for row in reader:
            study_id = (row.get(normalized["studyinstanceuid"]) or "").strip()
            series_id = (row.get(normalized["seriesinstanceuid"]) or "").strip()
            if study_id not in requested or not series_id:
                continue
            key = (study_id, series_id)
            if key in seen:
                raise ValueError(f"I found duplicate study/series manifest entry: {key}")
            seen.add(key)
            candidates[study_id].append(
                {
                    "SeriesInstanceUID": series_id,
                    "Fluid_Sensitive": (
                        row.get(normalized["fluid_sensitive"]) or ""
                    ).strip(),
                    "Fat_Suppression": (
                        row.get(normalized["fat_suppression"]) or ""
                    ).strip(),
                    "Anatomical_Plane": (
                        row.get(normalized["anatomical_plane"]) or ""
                    ).strip(),
                }
            )
    return candidates
