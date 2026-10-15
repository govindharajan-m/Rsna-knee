"""I test the smallest trainable MRI baseline without competition data."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from kneescope12.data import volume as volume_module
from kneescope12.data.dataset import BaselineStudyDataset
from kneescope12.data.intensity import normalize_mri_volume
from kneescope12.data.preprocessing import preprocess_baseline_volume
from kneescope12.data.resampling import resample_to_shape
from kneescope12.data.volume import reconstruct_ordered_volume
from kneescope12.model import BaselineClassifier
from scripts.predict_baseline import generate_submission, validate_submission

LABEL_COLUMNS = [f"finding_{index}" for index in range(12)]


def _load_training_script():
    script = Path(__file__).parents[1] / "scripts" / "train_baseline.py"
    spec = importlib.util.spec_from_file_location("train_baseline", script)
    if spec is None or spec.loader is None:
        raise AssertionError(f"I could not load baseline trainer from {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_pixel_dicom(
    path: Path,
    values: np.ndarray,
    *,
    position_z: float,
    study_uid: str,
    series_uid: str,
    instance_number: int,
    orientation: tuple[float, float, float, float, float, float] = (
        1,
        0,
        0,
        0,
        1,
        0,
    ),
) -> Path:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.StudyInstanceUID = study_uid
    dataset.SeriesInstanceUID = series_uid
    dataset.Modality = "MR"
    dataset.Rows, dataset.Columns = values.shape
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.PixelSpacing = [0.7, 0.7]
    dataset.SliceThickness = 3.0
    dataset.ImageOrientationPatient = list(orientation)
    dataset.ImagePositionPatient = [0, 0, position_z]
    dataset.InstanceNumber = instance_number
    dataset.PixelData = np.asarray(values, dtype=np.uint16).tobytes()
    dataset.save_as(path, enforce_file_format=True)
    return path


def test_volume_reconstruction_uses_geometry_order_and_checks_slice_count(
    tmp_path, monkeypatch
):
    study_uid = generate_uid()
    series_uid = generate_uid()
    first = _write_pixel_dicom(
        tmp_path / "slice-z.dcm",
        np.full((4, 4), 30),
        position_z=3,
        study_uid=study_uid,
        series_uid=series_uid,
        instance_number=1,
    )
    second = _write_pixel_dicom(
        tmp_path / "slice-a.dcm",
        np.full((4, 4), 10),
        position_z=1,
        study_uid=study_uid,
        series_uid=series_uid,
        instance_number=2,
    )
    volume = reconstruct_ordered_volume([first, second])

    assert volume.shape == (2, 4, 4)
    assert volume.dtype == np.float32
    assert volume[:, 0, 0].tolist() == [10.0, 30.0]

    monkeypatch.setattr(
        volume_module,
        "order_dicom_slices",
        lambda paths: type("Ordering", (), {"slices": (object(),)})(),
    )
    with pytest.raises(ValueError, match="metadata slice count"):
        reconstruct_ordered_volume([first, second])


def test_normalization_handles_nonfinite_constant_and_all_invalid_volumes():
    normalized = normalize_mri_volume(
        np.asarray([[-np.inf, 0.0, 1.0], [100.0, np.nan, np.inf]], dtype=np.float32)
    )
    assert np.isfinite(normalized).all()
    assert normalized.min() == 0.0
    assert normalized.max() == 1.0
    assert normalized[0, 0] == 0.0
    assert normalized[1, 1] == 0.0
    assert np.count_nonzero(normalize_mri_volume(np.full((2, 2), 4.0))) == 0
    assert np.count_nonzero(normalize_mri_volume(np.full((2, 2), np.nan))) == 0


def test_resampling_shape_and_preprocessing_are_deterministic():
    source = np.arange(4 * 8 * 10, dtype=np.float32).reshape(4, 8, 10)
    resized = resample_to_shape(source, (6, 12, 14), order=1)
    first = preprocess_baseline_volume(
        source,
        target_shape=(6, 12, 14),
        lower_percentile=1,
        upper_percentile=99,
    )
    second = preprocess_baseline_volume(
        source,
        target_shape=(6, 12, 14),
        lower_percentile=1,
        upper_percentile=99,
    )

    assert resized.shape == (6, 12, 14)
    assert resized.dtype == np.float32
    assert first.shape == (6, 12, 14)
    assert np.array_equal(first, second)
    assert 0 <= float(first.min()) <= float(first.max()) <= 1


def test_dataset_returns_fixed_plane_channels_and_twelve_explicit_labels(
    tmp_path, monkeypatch
):
    dataset_root = tmp_path / "data"
    study_id = "study-1"
    series_rows = [
        ("Sagittal", "series-sag"),
        ("Coronal", "series-cor"),
        ("Axial", "series-axial-late"),
        ("Axial", "series-axial-first"),
    ]
    for _plane, series_id in series_rows:
        directory = dataset_root / "train_series" / study_id / series_id
        directory.mkdir(parents=True)
        (directory / "placeholder.dcm").write_bytes(b"synthetic")
    series_csv = dataset_root / "train_series.csv"
    with series_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "StudyInstanceUID",
                "SeriesInstanceUID",
                "Fluid_Sensitive",
                "Fat_Suppression",
                "Anatomical_Plane",
            ]
        )
        writer.writerows(
            [
                (study_id, "series-sag", "1", "0", "Sagittal"),
                (study_id, "series-cor", "0", "0", "Coronal"),
                (study_id, "series-axial-late", "1", "0", "Axial"),
                (study_id, "series-axial-first", "0", "1", "Axial"),
            ]
        )
    labels_csv = dataset_root / "train.csv"
    with labels_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["StudyInstanceUID", "Report", *LABEL_COLUMNS])
        writer.writerow([study_id, "ignored report", *([1] + [0] * 11)])

    selected_series = []

    def fake_reconstruct(paths):
        selected_series.append(paths[0].parent.name)
        return np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)

    monkeypatch.setattr(
        "kneescope12.data.dataset.reconstruct_ordered_volume", fake_reconstruct
    )
    dataset = BaselineStudyDataset(
        dataset_root,
        labels_csv,
        series_csv,
        [study_id],
        LABEL_COLUMNS,
        target_shape=(4, 8, 8),
        plane_order=["Axial", "Coronal", "Sagittal"],
        max_series_per_study=3,
    )
    image, target = dataset[0]

    assert image.shape == (3, 4, 8, 8)
    assert image.dtype == torch.float32
    assert target.shape == (12,)
    assert target.tolist() == [1.0] + [0.0] * 11
    assert selected_series == ["series-axial-first", "series-cor", "series-sag"]


def test_small_classifier_returns_twelve_logits_and_bce_loss_backpropagates():
    model = BaselineClassifier(in_channels=3, base_channels=4)
    images = torch.randn(2, 3, 8, 16, 16)
    targets = torch.randint(0, 2, (2, 12), dtype=torch.float32)
    logits = model(images)
    loss = torch.nn.BCEWithLogitsLoss()(logits, targets)
    loss.backward()

    assert logits.shape == (2, 12)
    assert torch.isfinite(loss)
    assert model.classifier.weight.grad is not None


def test_validation_auc_reports_only_calculable_labels():
    trainer = _load_training_script()
    targets = np.asarray([[0, 1], [1, 1], [0, 1], [1, 1]], dtype=np.float32)
    logits = np.asarray([[0, 1], [1, 1], [1, 1], [2, 1]], dtype=np.float32)

    report = trainer._auc_report(targets, logits, ["varying", "all-positive"])

    assert report["per_label"]["varying"]["auc"] == pytest.approx(0.875)
    assert report["per_label"]["varying"]["status"] == "calculated"
    assert report["per_label"]["all-positive"]["auc"] is None
    assert report["per_label"]["all-positive"]["positive_count"] == 4
    assert report["per_label"]["all-positive"]["negative_count"] == 0
    assert report["macro_auc"] == pytest.approx(0.875)
    assert report["calculable_label_count"] == 1


def test_inference_generates_probability_submission_matching_official_template(
    tmp_path,
):
    dataset_root = tmp_path / "dataset"
    study_id = generate_uid()
    orientations = {
        "Axial": (1, 0, 0, 0, 1, 0),
        "Coronal": (1, 0, 0, 0, 0, 1),
        "Sagittal": (0, 1, 0, 0, 0, 1),
    }
    for plane in ("Axial", "Coronal", "Sagittal"):
        series_id = generate_uid()
        directory = dataset_root / "test_series" / study_id / series_id
        directory.mkdir(parents=True)
        for index, position in enumerate((0.0, 3.0), start=1):
            _write_pixel_dicom(
                directory / f"slice-{index}.dcm",
                np.full((8, 8), index, dtype=np.uint16),
                position_z=position,
                study_uid=study_id,
                series_uid=series_id,
                instance_number=index,
                orientation=orientations[plane],
            )
    sample_path = dataset_root / "sample_submission.csv"
    with sample_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["StudyInstanceUID", *LABEL_COLUMNS])
        writer.writerow([study_id, *([0] * 12)])

    baseline_config = {
        "baseline": {
            "max_series_per_study": 3,
            "plane_order": ["Axial", "Coronal", "Sagittal"],
            "target_shape": [4, 8, 8],
            "preprocessing": {
                "lower_percentile": 0.5,
                "upper_percentile": 99.5,
                "interpolation_order": 1,
            },
        },
        "model": {"base_channels": 2},
        "labels": LABEL_COLUMNS,
    }
    model = BaselineClassifier(in_channels=3, base_channels=2)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": baseline_config,
            "plane_axis_mapping": {"Axial": 2, "Coronal": 1, "Sagittal": 0},
        },
        checkpoint_path,
    )
    output_path = tmp_path / "submission.csv"
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        report = generate_submission(
            dataset_root,
            checkpoint_path,
            output_path,
            batch_size=1,
            device_name="cpu",
            series_csv=None,
        )
    finally:
        torch.set_num_threads(previous_threads)

    assert report["validation"]["valid"] is True
    assert report["validation"]["row_count"] == 1
    assert report["validation"]["prediction_column_count"] == 12
    assert validate_submission(sample_path, output_path)["study_id_order_matches"] is True
    with output_path.open(newline="", encoding="utf-8") as handle:
        output_rows = list(csv.DictReader(handle))
    predictions = [float(output_rows[0][column]) for column in LABEL_COLUMNS]
    assert all(np.isfinite(value) and 0 <= value <= 1 for value in predictions)


def test_training_smoke_runs_one_synthetic_dicom_batch(tmp_path):
    module = _load_training_script()

    dataset_root = tmp_path / "dataset"
    series_rows = []
    study_ids = [generate_uid() for _ in range(2)]
    orientations = {
        "Axial": (1, 0, 0, 0, 1, 0),
        "Coronal": (1, 0, 0, 0, 0, 1),
        "Sagittal": (0, 1, 0, 0, 0, 1),
    }
    for study_index in range(2):
        study_id = study_ids[study_index]
        for plane_index, plane in enumerate(("Axial", "Coronal", "Sagittal")):
            series_id = generate_uid()
            directory = dataset_root / "train_series" / study_id / series_id
            directory.mkdir(parents=True)
            for slice_index, position in enumerate((0.0, 3.0), start=1):
                values = np.arange(64, dtype=np.uint16).reshape(8, 8) + plane_index + slice_index
                _write_pixel_dicom(
                    directory / f"slice-{slice_index}.dcm",
                    values,
                    position_z=position,
                    study_uid=study_id,
                    series_uid=series_id,
                    instance_number=slice_index,
                    orientation=orientations[plane],
                )
            series_rows.append(
                (study_id, series_id, "1", "0", plane)
            )
    series_csv = dataset_root / "train_series.csv"
    with series_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "StudyInstanceUID",
                "SeriesInstanceUID",
                "Fluid_Sensitive",
                "Fat_Suppression",
                "Anatomical_Plane",
            ]
        )
        writer.writerows(series_rows)
    labels_csv = dataset_root / "train.csv"
    with labels_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["StudyInstanceUID", *LABEL_COLUMNS])
        writer.writerow([study_ids[0], *([0] * 12)])
        writer.writerow([study_ids[1], *([1] * 12)])

    config = {
        "seed": 17,
        "paths": {"dataset_root": None, "checkpoint": str(tmp_path / "model.pt")},
        "labels": LABEL_COLUMNS,
        "baseline": {
            "target_shape": [4, 8, 8],
            "plane_order": ["Axial", "Coronal", "Sagittal"],
            "max_series_per_study": 3,
            "preprocessing": {
                "lower_percentile": 0.5,
                "upper_percentile": 99.5,
                "interpolation_order": 1,
            },
        },
        "model": {"base_channels": 2},
        "training": {
            "batch_size": 1,
            "epochs": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
        },
        "validation": {"fraction": 0.5},
    }
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        report = module.run_training(
            SimpleNamespace(
                config=config_path,
                dataset_root=dataset_root,
                labels_csv=labels_csv,
                series_csv=series_csv,
                checkpoint=tmp_path / "smoke.pt",
                seed=None,
                epochs=None,
                batch_size=None,
                learning_rate=None,
                device="cpu",
                max_train_batches=None,
                smoke_test=True,
                train_all=False,
            )
        )
    finally:
        torch.set_num_threads(previous_threads)

    assert report["gold_labeled_study_count"] == 2
    assert report["train_study_count"] == 1
    assert report["validation_study_count"] == 1
    assert report["train_ids"][0] in study_ids
    assert report["history"][0]["logits_shape"] == [1, 12]
    assert np.isfinite(report["history"][0]["mean_train_loss"])
    assert report["checkpoint"] is None
    assert report["plane_axis_mapping"] == {
        "Axial": 2,
        "Coronal": 1,
        "Sagittal": 0,
    }
