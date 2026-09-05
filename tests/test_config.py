"""I test explicit YAML configuration behavior."""

from __future__ import annotations

import pytest

from kneescope12.utils.config import (
    ConfigError,
    load_config,
    merge_config,
    resolve_runtime_paths,
    save_config,
)


def test_load_and_save_config(tmp_path):
    """I round-trip a mapping through YAML without hidden defaults."""

    source = tmp_path / "input.yaml"
    source.write_text("seed: 7\npaths:\n  dataset_root: /data\n", encoding="utf-8")
    loaded = load_config(source)
    destination = save_config(loaded, tmp_path / "output.yaml")
    assert load_config(destination) == loaded


def test_merge_config_does_not_mutate_inputs():
    """I preserve nested base values while applying nested overrides."""

    base = {"paths": {"dataset_root": "a", "audit_output": "b"}, "seed": 1}
    merged = merge_config(base, {"paths": {"dataset_root": "c"}})
    assert merged == {"paths": {"dataset_root": "c", "audit_output": "b"}, "seed": 1}
    assert base["paths"]["dataset_root"] == "a"


def test_load_rejects_non_mapping_yaml(tmp_path):
    """I reject list roots because the project configuration must be keyed."""

    source = tmp_path / "invalid.yaml"
    source.write_text("- not\n- a mapping\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="root must be a mapping"):
        load_config(source)


def test_resolve_runtime_paths_uses_environment_overrides(monkeypatch, tmp_path):
    """I override configured project paths from explicit environment variables."""

    monkeypatch.setenv("KNEESCOPE_DATA_ROOT", "/mnt/kneescope/data")
    monkeypatch.setenv("KNEESCOPE_CACHE_ROOT", "/mnt/kneescope/cache")
    monkeypatch.setenv("KNEESCOPE_EXPERIMENT_ROOT", "/mnt/kneescope/experiments")
    monkeypatch.setenv("KNEESCOPE_CHECKPOINT_ROOT", "/mnt/kneescope/checkpoints")
    monkeypatch.setenv("KNEESCOPE_SUBMISSION_ROOT", "/mnt/kneescope/submissions")

    source = tmp_path / "config.yaml"
    source.write_text(
        """
paths:
  dataset_root: ${KNEESCOPE_DATA_ROOT}
  cache_root: ${KNEESCOPE_CACHE_ROOT}
  experiment_root: ${KNEESCOPE_EXPERIMENT_ROOT}
  checkpoint_root: ${KNEESCOPE_CHECKPOINT_ROOT}
  submission_root: ${KNEESCOPE_SUBMISSION_ROOT}
""".strip(),
        encoding="utf-8",
    )

    resolved = resolve_runtime_paths(load_config(source), project_root=tmp_path)
    assert resolved["paths"]["dataset_root"] == "/mnt/kneescope/data"
    assert resolved["paths"]["cache_root"] == "/mnt/kneescope/cache"
    assert resolved["paths"]["experiment_root"] == "/mnt/kneescope/experiments"
    assert resolved["paths"]["checkpoint_root"] == "/mnt/kneescope/checkpoints"
    assert resolved["paths"]["submission_root"] == "/mnt/kneescope/submissions"
