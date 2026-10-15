# KneeScope-12

KneeScope-12 is my experimental framework for twelve-class knee MRI abnormality detection in the RSNA knee MRI abnormality detection competition.

## Research objective

I am investigating whether a combination of anatomical localization, multi-resolution MRI representations, finding-specific attention, and probabilistic weak supervision can improve MRI-based detection of the twelve competition findings. I have not established comparative performance, novelty, or superiority.

## Current status

> A small MRI-only training baseline is implemented for the explicitly labeled
> studies. It is an engineering baseline, not a validated competition model.

The baseline provides strict geometry-ordered volume reconstruction, fixed-shape
preprocessing, a compact 3D CNN, and a one-batch smoke-test mode. The training
script does not use reports, weak labels, or ROI localization; a separate
inference script and the Kaggle notebook generate test probabilities.

## Baseline data policy

```text
DICOM series -> physical slice ordering -> [depth, row, column] volume
  -> per-volume percentile scaling -> fixed-shape resize
  -> plane channels -> small 3D CNN -> 12 logits
```

I use only rows with all twelve configured binary labels in `train.csv`; unlabeled
report rows are excluded. For each manifest plane, I select the lexicographically
first `SeriesInstanceUID`. Channels are fixed as Axial, Coronal, Sagittal; an
absent plane is zero-filled. This is a reproducible candidate rule, not an
optimal or medically validated series choice. Missing or malformed selected data
raises an explicit error.

If Kaggle supplies no test-series manifest, the notebook derives a one-to-one
association between training manifest plane names and dominant DICOM slice-normal
coordinate axes from the selected gold-training series. It applies that measured
association to test series metadata and selects the lowest UID for each plane.
This is an empirical dataset mapping, not an L/R or A/P interpretation.

I order slices with the existing physical-geometry ordering. I retain the DICOM
acquisition's in-plane row/column direction and resize the index extent to the
configured `[depth, row, column]` shape. I do not canonicalize patient orientation
or claim a target physical voxel spacing. Intensity values are clipped at
configurable per-volume percentiles and scaled to `[0, 1]`; non-finite values
become zero. These settings are computationally convenient starting points, not
scientifically selected MRI preprocessing.

The dataset keeps processed tensors in an in-process per-study cache to avoid
re-decoding selected DICOM series on each epoch. Cache memory therefore grows
with the number of studies visited.

## Running the baseline

I configure dataset paths and preprocessing/model/training parameters in
`configs/baseline.yaml`, or override paths and hyperparameters on the command
line. I can verify one real training batch, including preprocessing, logits,
loss, and backward pass, with:

```bash
python scripts/train_baseline.py \
  --config configs/baseline.yaml \
  --dataset-root /path/to/rsna-knee-abnormality-detection \
  --smoke-test
```

I train for the configured epochs and write a checkpoint outside source modules:

```bash
python scripts/train_baseline.py \
  --config configs/baseline.yaml \
  --dataset-root /path/to/rsna-knee-abnormality-detection
```

The split is a seeded study-level random split, not stratified; with only 58
labeled studies, per-finding validation ROC AUC is reported only when the
validation subset contains both positive and negative examples. Labels with
insufficient examples are explicitly marked as unavailable. The Kaggle notebook
then refits the same small model using all explicitly labeled studies and
generates a probability CSV in sample-submission row/column order. The inference
script validates every row and prediction before reporting success.

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

The package adds only the data and model components required for this baseline.
Report processing, weak supervision, and ROI localization remain future work.

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
