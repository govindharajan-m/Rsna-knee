"""I load and serialize explicit YAML configuration mappings."""

from __future__ import annotations

import os
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """I signal invalid or unusable configuration content."""


Config = dict[str, Any]

_PATH_ENV_VARS: dict[str, str] = {
    "dataset_root": "KNEESCOPE_DATA_ROOT",
    "cache_root": "KNEESCOPE_CACHE_ROOT",
    "experiment_root": "KNEESCOPE_EXPERIMENT_ROOT",
    "checkpoint_root": "KNEESCOPE_CHECKPOINT_ROOT",
    "submission_root": "KNEESCOPE_SUBMISSION_ROOT",
    "audit_output": "KNEESCOPE_AUDIT_OUTPUT",
    "manifest_output": "KNEESCOPE_MANIFEST_OUTPUT",
    "qc_output": "KNEESCOPE_QC_OUTPUT",
}


def load_config(path: str | Path) -> Config:
    """I load a YAML mapping from *path* and reject non-mapping root values."""

    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConfigError(f"Configuration file does not exist: {config_path}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"Configuration YAML is invalid: {config_path}") from error

    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"Configuration root must be a mapping: {config_path}")
    return raw


def merge_config(base: Mapping[str, Any], override: Mapping[str, Any]) -> Config:
    """I recursively merge an override mapping without mutating either input."""

    merged: Config = deepcopy(dict(base))
    for key, value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, Mapping) and isinstance(value, Mapping):
            merged[key] = merge_config(base_value, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def save_config(config: Mapping[str, Any], path: str | Path) -> Path:
    """I serialize a configuration mapping as stable, human-readable YAML."""

    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(dict(config), sort_keys=True, allow_unicode=True), encoding="utf-8"
    )
    return config_path


def resolve_runtime_paths(config: Mapping[str, Any], *, project_root: str | Path) -> Config:
    """I resolve runtime paths with explicit environment overrides and config templates."""

    resolved = deepcopy(dict(config))
    if not isinstance(resolved.get("paths", {}), Mapping):
        raise ConfigError("paths configuration must be a mapping")
    path_values = dict(resolved["paths"])
    for key, value in path_values.items():
        env_name = _PATH_ENV_VARS.get(key)
        if env_name is not None and os.getenv(env_name):
            path_values[key] = os.getenv(env_name)
        elif isinstance(value, str):
            path_values[key] = _expand_path_template(value, field_name=f"paths.{key}")
        elif value is not None:
            path_values[key] = value
    resolved["paths"] = path_values
    return resolved


def resolve_path(value: str | Path, *, project_root: str | Path) -> Path:
    """I resolve an absolute or project-relative path without consulting hidden state."""

    candidate = Path(_expand_path_template(str(value), field_name="path")).expanduser()
    if candidate.is_absolute():
        return candidate
    return Path(project_root) / candidate


def _expand_path_template(value: str, *, field_name: str) -> str:
    """I resolve environment templates in a path string while rejecting unresolved variables."""

    expanded = os.path.expandvars(value)
    if "${" in expanded or "$" in expanded:
        unresolved = [token for token in _iter_env_tokens(expanded) if token]
        if unresolved:
            raise ConfigError(f"Unresolved environment variable in {field_name}: {value}")
    return expanded


def _iter_env_tokens(value: str) -> list[str]:
    """I recover environment-variable tokens used in a string value."""

    tokens: list[str] = []
    for token in value.split():
        if "$" in token:
            tokens.append(token)
    return tokens
