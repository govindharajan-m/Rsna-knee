> I configure the data root through environment variables instead of embedding a dataset path in source code.

# Cloud GPU setup for KneeScope-12

I use this repository in a standard Linux GPU cloud VM. I keep the project portable and provider-agnostic: the machine mounts persistent disk, exposes an NVIDIA GPU, and runs Python 3.11 with CUDA-compatible PyTorch.

## 1. Create the environment

I create a fresh environment from the repository root:

```bash
cd /workspace/kneescope12
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

I can also use the Conda workflow:

```bash
conda env create -f environment.yml
conda activate kneescope12
pip install -e .[dev]
```

## 2. Configure runtime paths

I keep all large datasets, caches, experiment outputs, checkpoints, and submission artifacts outside the Git tree. I configure them through environment variables:

```bash
export KNEESCOPE_DATA_ROOT=/mnt/data/kneescope12
export KNEESCOPE_CACHE_ROOT=/mnt/cache/kneescope12
export KNEESCOPE_EXPERIMENT_ROOT=/mnt/experiments/kneescope12
export KNEESCOPE_CHECKPOINT_ROOT=/mnt/checkpoints/kneescope12
export KNEESCOPE_SUBMISSION_ROOT=/mnt/submissions/kneescope12
```

The default YAML config in `configs/baseline.yaml` can also be set to these values, but I prefer environment variables because they are easy to export in a cloud terminal and persist in shell profiles.

## 3. Install dependencies and verify GPU compatibility

I verify the GPU-visible environment before running preprocessing:

```bash
python scripts/check_environment.py
```

I expect:

- Python 3.11
- a CUDA-capable PyTorch build
- `torch.cuda.is_available()` true
- `torch.version.cuda` matching the system CUDA runtime
- a detected NVIDIA GPU name and memory

If CUDA is missing, I fix the environment before proceeding. I do not require the model to run yet; the pipeline only needs a verified GPU-visible runtime and compatible PyTorch version.

## 4. Mount or provide the competition dataset

I do not automatically download the competition dataset. I mount or copy the official competition dataset into `DATA_ROOT` manually.

Typical setup:

```bash
mkdir -p "$KNEESCOPE_DATA_ROOT"
# mount a persistent disk or copy the competition files here
ls -lh "$KNEESCOPE_DATA_ROOT"
```

I do not store Kaggle credentials, cloud credentials, SSH keys, or passwords in the repository. If I use the Kaggle CLI, I keep the token in the cloud user home or a secure secret manager and never commit it.

## 5. Run a tiny smoke test

I test the full pipeline on a tiny subset without touching the full competition dataset:

```bash
python scripts/audit_dataset.py --data-root "$KNEESCOPE_DATA_ROOT" --output-dir "$KNEESCOPE_EXPERIMENT_ROOT/dataset_audit" --limit 10
```

This verifies:

- the cloud environment is healthy
- DICOM metadata reads are working
- output directories are writable
- logs are created
- failures are handled cleanly
- cache and manifest output paths are valid

## 6. Run the full audit

I run the full metadata audit once the dataset is mounted:

```bash
python scripts/audit_dataset.py --data-root "$KNEESCOPE_DATA_ROOT" --output-dir "$KNEESCOPE_EXPERIMENT_ROOT/dataset_audit"
```

This writes JSON and CSV summaries into the configured output directory.

## 7. Run resumable preprocessing

I use the preprocessing pipeline with a persistent cache and manifest path:

```bash
python scripts/prepare_dataset.py \
  --data-root "$KNEESCOPE_DATA_ROOT" \
  --cache-root "$KNEESCOPE_CACHE_ROOT" \
  --manifest-path "$KNEESCOPE_EXPERIMENT_ROOT/manifests/study_manifest.parquet" \
  --output-dir "$KNEESCOPE_EXPERIMENT_ROOT/preprocessing" \
  --config configs/baseline.yaml
```

The cache is versioned by preprocessing configuration and source fingerprint. If I rerun the same pipeline on the same data after a disruption, completed studies are skipped safely. If the preprocessing configuration changes, the system will not silently reuse stale outputs.

## 8. Locate logs and outputs

I keep logs under the configured experiment root:

```bash
ls -R "$KNEESCOPE_EXPERIMENT_ROOT"
```

The audit logs are placed next to the audit output directory. The cache, manifest, QC outputs, and structured reports stay outside Git and outside the source tree.

## 9. Shut down and restart safely

I do not assume the VM will always stay up. I use a persistent disk and maintain a configuration script for the environment so I can restart cleanly:

```bash
source ~/.bashrc
export KNEESCOPE_DATA_ROOT=/mnt/data/kneescope12
export KNEESCOPE_CACHE_ROOT=/mnt/cache/kneescope12
export KNEESCOPE_EXPERIMENT_ROOT=/mnt/experiments/kneescope12
export KNEESCOPE_CHECKPOINT_ROOT=/mnt/checkpoints/kneescope12
export KNEESCOPE_SUBMISSION_ROOT=/mnt/submissions/kneescope12
```

For long-running jobs, I use `tmux`, `screen`, `nohup`, or a provider-managed persistent job mechanism depending on the machine. I do not require a specific tool.

## 10. Cloud job safety

I keep the data pipeline resilient to interruptions:

- raw DICOM files are read incrementally
- large datasets are not loaded into RAM
- preprocessing writes atomic cache artifacts
- failed studies are logged without stopping the entire job
- progress is reported through logs
- already processed studies are skipped when the configuration and source are unchanged

## 11. Next step

I will only add model training after the environment is verified and the data pipeline is stable on the cloud GPU host.
