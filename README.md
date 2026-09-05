# KneeScope-12

KneeScope-12 is my experimental framework for twelve-class knee MRI abnormality detection in the RSNA knee MRI abnormality detection competition.

## Research objective

I am investigating whether a combination of anatomical localization, multi-resolution MRI representations, finding-specific attention, and probabilistic weak supervision can improve MRI-based detection of the twelve competition findings. I have not established comparative performance, novelty, or superiority.

## Current status

> Foundation and dataset-audit infrastructure under development.

I have not implemented a neural network. This repository currently provides configuration, reproducibility, DICOM metadata access, and a dataset-audit command only.

## Planned pipeline

```text
MRI / DICOM
  -> quality control
  -> series identification
  -> volume construction
  -> anatomical localization
  -> global / local representations
  -> feature extraction
  -> finding-specific attention
  -> twelve outputs
```

```text
Radiology report
  -> language / finding analysis
  -> uncertainty and negation handling
  -> probabilistic weak labels
  -> calibration
  -> training supervision
```

I intend to use radiology reports only for training-time supervision. My final inference pipeline must operate from permitted MRI/test inputs and must not depend on unavailable test reports.

## Experimental philosophy

I will use controlled experiments, cross-validation, ablation studies, reproducible configurations, and explicit experiment records. I will record the configuration, source revision, random seed, environment, and dataset version for each future training experiment.

## Repository layout

```text
configs/              YAML experiment and audit settings
src/kneescope12/      Installable Python package
scripts/              Command-line entry points
tests/                Synthetic-data unit tests
experiments/          Ignored local experiment outputs
submissions/          Ignored local submission outputs
notebooks/            Optional exploratory notebooks
```

The package reserves `data`, `reports`, `models`, `training`, `inference`, and `utils` modules for incremental development. The model, report processing, training, and submission components are intentionally absent at this stage.

## Quick start

I use Python 3.14.3. I create the environment and install the package in editable mode:

```bash
conda env create -f environment.yml
conda activate kneescope12
pip install -e ".[dev]"
pytest
```

I can inspect the audit interface without a dataset:

```bash
python scripts/audit_dataset.py --help
```

I audit a configured or explicit dataset location with:

```bash
python scripts/audit_dataset.py --dataset-root /path/to/dataset --output-dir experiments/dataset_audit
```

The audit reads metadata without loading image pixels, records malformed candidate files in its log, and writes `audit_summary.json` plus `series_inventory.csv`. I configure dataset paths rather than assuming a competition filesystem layout.

## Data and artifacts

I do not commit competition data, DICOM files, model weights, checkpoints, credentials, or large generated artifacts. I store large artifacts in approved external storage or a dataset/artifact system, then record their source, version, access conditions, and compliance status in the registry documents before use.

## Compliance and licenses

I track unverified competition requirements in [COMPLIANCE.md](COMPLIANCE.md) and external resource licenses in [LICENSES.md](LICENSES.md). I will verify requirements against the current official competition rules before using external resources or making a submission.
