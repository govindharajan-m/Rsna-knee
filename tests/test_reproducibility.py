"""I test deterministic seed and provenance utilities."""

from __future__ import annotations

import json
import random
import re
from datetime import UTC, datetime

import numpy as np

from kneescope12.utils.experiments import generate_experiment_id, write_experiment_record
from kneescope12.utils.reproducibility import set_global_seed


def test_set_global_seed_repeats_python_and_numpy_sequences():
    """I verify that repeated explicit seeds reproduce local random sequences."""

    set_global_seed(17)
    first = (random.random(), float(np.random.random()))
    set_global_seed(17)
    second = (random.random(), float(np.random.random()))
    assert first == second


def test_experiment_id_has_sortable_traceable_parts():
    """I verify that an experiment identifier includes time, label, and seed."""

    identifier = generate_experiment_id("Baseline QC", 17, now=datetime(2026, 1, 2, tzinfo=UTC))
    assert re.fullmatch(r"20260102T000000Z_baseline-qc_s17_[0-9a-f]{8}", identifier)


def test_experiment_record_captures_config_and_provenance(tmp_path):
    """I write the three provenance records that later experiments require."""

    output = write_experiment_record(
        tmp_path / "record",
        config={"seed": 17},
        experiment_id="test-id",
        seed=17,
        dataset_version="synthetic-v1",
        project_root=tmp_path,
    )
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert (output / "config.yaml").is_file()
    assert (output / "environment.json").is_file()
    assert provenance["dataset_version"] == "synthetic-v1"
    assert provenance["git_revision"] is None
