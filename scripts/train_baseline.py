"""I train or smoke-test the small MRI-only 12-output baseline."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy.stats import rankdata
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from kneescope12.data.dataset import (  # noqa: E402, I001
    BaselineStudyDataset,
    derive_plane_axis_mapping,
    read_gold_labels,
)
from kneescope12.model import BaselineClassifier  # noqa: E402
from kneescope12.utils.reproducibility import set_global_seed  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--labels-csv", type=Path)
    parser.add_argument("--series-csv", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument(
        "--train-all",
        action="store_true",
        help="Fit all explicitly labeled studies after validation decisions are made",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one actual training batch and backward pass, then exit without checkpointing",
    )
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"I expected a YAML mapping in {path}")
    return config


def _split_studies(
    study_ids: list[str], *, seed: int, validation_fraction: float
) -> tuple[list[str], list[str]]:
    """I create a reproducible study-level split without risky 12-label stratification."""

    if len(study_ids) < 2:
        raise ValueError("I need at least two gold-labeled studies for a train/validation split")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be strictly between zero and one")
    permutation = np.random.default_rng(seed).permutation(len(study_ids))
    validation_count = min(
        len(study_ids) - 1,
        max(1, math.ceil(len(study_ids) * validation_fraction)),
    )
    validation_indices = set(int(value) for value in permutation[:validation_count])
    train_ids = [
        study_id
        for index, study_id in enumerate(study_ids)
        if index not in validation_indices
    ]
    validation_ids = [
        study_id for index, study_id in enumerate(study_ids) if index in validation_indices
    ]
    return train_ids, validation_ids


def _dataset(
    config: dict[str, Any],
    dataset_root: Path,
    labels_csv: Path,
    series_csv: Path,
    study_ids: list[str],
    label_columns: list[str],
) -> BaselineStudyDataset:
    baseline = config["baseline"]
    preprocessing = baseline["preprocessing"]
    return BaselineStudyDataset(
        dataset_root,
        labels_csv,
        series_csv,
        study_ids,
        label_columns,
        target_shape=tuple(int(value) for value in baseline["target_shape"]),
        plane_order=list(baseline["plane_order"]),
        max_series_per_study=int(baseline["max_series_per_study"]),
        lower_percentile=float(preprocessing["lower_percentile"]),
        upper_percentile=float(preprocessing["upper_percentile"]),
        interpolation_order=int(preprocessing["interpolation_order"]),
    )


def _auc_report(targets: np.ndarray, logits: np.ndarray, labels: list[str]) -> dict[str, Any]:
    """I calculate ROC AUC only where both binary classes are observed."""

    per_label: dict[str, dict[str, Any]] = {}
    calculable: list[float] = []
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
    for index, label in enumerate(labels):
        truth = targets[:, index].astype(np.int32)
        positives = int(np.sum(truth == 1))
        negatives = int(np.sum(truth == 0))
        if positives == 0 or negatives == 0:
            per_label[label] = {
                "auc": None,
                "positive_count": positives,
                "negative_count": negatives,
                "status": "insufficient_positive_or_negative",
            }
            continue
        ranks = rankdata(probabilities[:, index], method="average")
        auc = float(
            (np.sum(ranks[truth == 1]) - positives * (positives + 1) / 2)
            / (positives * negatives)
        )
        per_label[label] = {
            "auc": auc,
            "positive_count": positives,
            "negative_count": negatives,
            "status": "calculated",
        }
        calculable.append(auc)
    return {
        "per_label": per_label,
        "macro_auc": float(np.mean(calculable)) if calculable else None,
        "calculable_label_count": len(calculable),
    }


def _evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, labels: list[str]
) -> dict[str, Any]:
    model.eval()
    targets: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    with torch.no_grad():
        for images, truth in loader:
            output = model(images.to(device))
            logits.append(output.cpu().numpy())
            targets.append(truth.numpy())
    if not targets:
        raise ValueError("I cannot evaluate an empty validation split")
    return _auc_report(np.concatenate(targets), np.concatenate(logits), labels)


def run_training(arguments: argparse.Namespace) -> dict[str, Any]:
    config = _load_config(arguments.config)
    paths_config = config["paths"]
    dataset_root = arguments.dataset_root or paths_config.get("dataset_root")
    if dataset_root is None:
        raise ValueError("I need --dataset-root or paths.dataset_root in the config")
    dataset_root = Path(dataset_root).expanduser().resolve()
    labels_csv = arguments.labels_csv or paths_config.get("labels_csv")
    series_csv = arguments.series_csv or paths_config.get("series_csv")
    labels_csv = Path(labels_csv) if labels_csv else dataset_root / "train.csv"
    series_csv = Path(series_csv) if series_csv else dataset_root / "train_series.csv"

    training = config["training"]
    baseline = config["baseline"]
    seed = arguments.seed if arguments.seed is not None else int(config["seed"])
    batch_size = (
        arguments.batch_size
        if arguments.batch_size is not None
        else int(training["batch_size"])
    )
    epochs = arguments.epochs if arguments.epochs is not None else int(training["epochs"])
    learning_rate = (
        arguments.learning_rate
        if arguments.learning_rate is not None
        else float(training["learning_rate"])
    )
    if batch_size <= 0 or epochs <= 0 or learning_rate <= 0:
        raise ValueError("I require positive batch size, epochs, and learning rate")
    if arguments.max_train_batches is not None and arguments.max_train_batches <= 0:
        raise ValueError("--max-train-batches must be positive")
    set_global_seed(seed, deterministic=True)
    label_columns = list(config["labels"])
    labels_by_study = read_gold_labels(labels_csv, label_columns)
    study_ids = sorted(labels_by_study)
    plane_axis_mapping = None
    plane_axis_counts = None
    plane_axis_mapping_error = None
    try:
        plane_axis_mapping, plane_axis_counts = derive_plane_axis_mapping(
            dataset_root,
            series_csv,
            study_ids,
            list(baseline["plane_order"]),
        )
    except ValueError as error:
        plane_axis_mapping_error = str(error)
    if arguments.train_all:
        train_ids, validation_ids = study_ids, []
    else:
        train_ids, validation_ids = _split_studies(
            study_ids,
            seed=seed,
            validation_fraction=float(config["validation"]["fraction"]),
        )
    train_dataset = _dataset(
        config, dataset_root, labels_csv, series_csv, train_ids, label_columns
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if arguments.device == "auto"
        else torch.device(arguments.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("I was asked to use CUDA, but CUDA is unavailable")
    model = BaselineClassifier(
        in_channels=int(baseline["max_series_per_study"]),
        base_channels=int(config["model"]["base_channels"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(training["weight_decay"]),
    )
    criterion = nn.BCEWithLogitsLoss()
    validation_loader = None
    if validation_ids:
        validation_dataset = _dataset(
            config, dataset_root, labels_csv, series_csv, validation_ids, label_columns
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
    checkpoint = arguments.checkpoint or Path(paths_config["checkpoint"])
    checkpoint = Path(checkpoint).expanduser()
    result: dict[str, Any] = {
        "seed": seed,
        "device": str(device),
        "gold_labeled_study_count": len(study_ids),
        "train_study_count": len(train_ids),
        "validation_study_count": len(validation_ids),
        "train_ids": train_ids,
        "validation_ids": validation_ids,
        "target_tensor_shape": [
            int(baseline["max_series_per_study"]),
            *[int(value) for value in baseline["target_shape"]],
        ],
        "plane_axis_mapping": plane_axis_mapping,
        "plane_axis_counts": plane_axis_counts,
        "plane_axis_mapping_error": plane_axis_mapping_error,
        "history": [],
        "checkpoint": None,
    }

    for epoch in range(epochs):
        model.train()
        batch_losses: list[float] = []
        last_logits_shape: list[int] | None = None
        for batch_index, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(images)
            if output.shape != targets.shape or output.shape[-1] != 12:
                raise RuntimeError(
                    f"I expected matching 12-logit output and targets, got "
                    f"{tuple(output.shape)} and {tuple(targets.shape)}"
                )
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
            last_logits_shape = list(output.shape)
            if arguments.smoke_test:
                result["history"].append(
                    {
                        "epoch": epoch + 1,
                        "train_batch_count": 1,
                        "mean_train_loss": batch_losses[0],
                        "logits_shape": last_logits_shape,
                    }
                )
                return result
            if (
                arguments.max_train_batches is not None
                and batch_index + 1 >= arguments.max_train_batches
            ):
                break
        train_metrics: dict[str, Any] = {
            "epoch": epoch + 1,
            "train_batch_count": len(batch_losses),
            "mean_train_loss": float(np.mean(batch_losses)),
            "logits_shape": last_logits_shape,
        }
        if validation_loader is not None:
            train_metrics["validation"] = _evaluate(
                model, validation_loader, device, label_columns
            )
        else:
            train_metrics["validation"] = None
            train_metrics["fit_scope"] = "all_explicitly_labeled_studies_no_validation"
        result["history"].append(train_metrics)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "config": config,
                "train_ids": train_ids,
                "validation_ids": validation_ids,
                "plane_axis_mapping": plane_axis_mapping,
                "plane_axis_counts": plane_axis_counts,
                "plane_axis_mapping_error": plane_axis_mapping_error,
                "validation_metrics": train_metrics["validation"],
            },
            checkpoint,
        )
        result["checkpoint"] = str(checkpoint)
    return result


def main() -> int:
    arguments = _parse_args()
    report = run_training(arguments)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
