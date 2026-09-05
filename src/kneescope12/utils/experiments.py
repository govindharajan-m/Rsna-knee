"""I capture experiment provenance before model development begins."""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .config import save_config
from .logging import get_logger

logger = get_logger("experiments")

_TRACKED_PACKAGES = ("numpy", "pydicom", "PyYAML", "kneescope12")


def generate_experiment_id(label: str, seed: int, *, now: datetime | None = None) -> str:
    """I create a sortable, collision-resistant identifier for an experiment record."""

    clean_label = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-") or "experiment"
    timestamp = (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{clean_label}_s{seed}_{uuid.uuid4().hex[:8]}"


def get_git_revision(project_root: str | Path) -> str | None:
    """I return the checked-out Git commit when the project is in a Git worktree."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(project_root),
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def collect_environment() -> dict[str, Any]:
    """I collect a compact, JSON-serializable runtime description."""

    packages: dict[str, str | None] = {}
    for package in _TRACKED_PACKAGES:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
    }


def write_experiment_record(
    directory: str | Path,
    *,
    config: Mapping[str, Any],
    experiment_id: str,
    seed: int,
    dataset_version: str | None,
    project_root: str | Path,
) -> Path:
    """I write config, environment, and source-version provenance to one directory."""

    record_dir = Path(directory)
    record_dir.mkdir(parents=True, exist_ok=False)
    save_config(config, record_dir / "config.yaml")
    (record_dir / "environment.json").write_text(
        json.dumps(collect_environment(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    provenance = {
        "experiment_id": experiment_id,
        "seed": seed,
        "dataset_version": dataset_version,
        "git_revision": get_git_revision(project_root),
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    (record_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    logger.info("I wrote experiment provenance to %s", record_dir)
    return record_dir


def require_string_sequence(value: Any, *, field_name: str) -> Sequence[str]:
    """I validate a sequence of strings for configuration adapters."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{field_name} must be a sequence of strings")
    if not all(isinstance(item, str) for item in value):
        raise TypeError(f"{field_name} must be a sequence of strings")
    return value
