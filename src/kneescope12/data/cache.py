"""I cache processed MRI arrays outside Git with explicit source and config invalidation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .preprocessing import ProcessedVolume, PreprocessingConfig


@dataclass(frozen=True)
class CachedVolume:
    """I expose one valid cache entry without hiding its provenance metadata."""

    global_volume: np.ndarray
    local_volume: np.ndarray
    metadata: dict[str, Any]
    cache_path: Path


class PreprocessingCache:
    """I store deterministic preprocessing artifacts under version and config fingerprints."""

    def __init__(self, root: str | Path, config: PreprocessingConfig) -> None:
        """I prepare a cache namespace for one preprocessing configuration."""

        self.root = Path(root)
        self.config = config
        self.namespace = self.root / config.version / config.fingerprint()

    def source_signature(self, paths: tuple[Path, ...]) -> str:
        """I fingerprint ordered source files using path, size, and modification time."""

        digest = hashlib.sha256()
        for path in paths:
            stat = path.stat()
            digest.update(str(path.resolve()).encode("utf-8"))
            digest.update(str(stat.st_size).encode("ascii"))
            digest.update(str(stat.st_mtime_ns).encode("ascii"))
        return digest.hexdigest()

    def cache_path(self, *, study_id: str | None, series_id: str | None) -> Path:
        """I create a deterministic, filesystem-safe path from non-identifying UIDs."""

        study_key = _identifier_key(study_id or "missing-study")
        series_key = _identifier_key(series_id or "missing-series")
        return self.namespace / study_key / f"{series_key}.npz"

    def load(
        self,
        *,
        study_id: str | None,
        series_id: str | None,
        source_signature: str,
    ) -> CachedVolume | None:
        """I return a cache entry only when source and preprocessing fingerprints both match."""

        path = self.cache_path(study_id=study_id, series_id=series_id)
        metadata_path = path.with_suffix(".json")
        if not path.is_file() or not metadata_path.is_file():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("preprocessing_fingerprint") != self.config.fingerprint():
                return None
            if metadata.get("source_signature") != source_signature:
                return None
            with np.load(path, allow_pickle=False) as archive:
                return CachedVolume(
                    global_volume=np.asarray(archive["global_volume"]),
                    local_volume=np.asarray(archive["local_volume"]),
                    metadata=metadata,
                    cache_path=path,
                )
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return None

    def store(
        self,
        *,
        study_id: str | None,
        series_id: str | None,
        source_signature: str,
        processed: ProcessedVolume,
        extra_metadata: Mapping[str, Any],
    ) -> CachedVolume:
        """I atomically write arrays and provenance for a deterministic cache entry."""

        path = self.cache_path(study_id=study_id, series_id=series_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "preprocessing_version": self.config.version,
            "preprocessing_fingerprint": self.config.fingerprint(),
            "source_signature": source_signature,
            "processing": processed.metadata_record(),
            **dict(extra_metadata),
        }
        _atomic_save_npz(
            path,
            global_volume=np.asarray(processed.global_volume),
            local_volume=np.asarray(processed.local_volume),
        )
        _atomic_write_text(
            path.with_suffix(".json"), json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        return CachedVolume(
            global_volume=np.asarray(processed.global_volume),
            local_volume=np.asarray(processed.local_volume),
            metadata=metadata,
            cache_path=path,
        )


def _identifier_key(value: str) -> str:
    """I hash a UID for a cache path without exposing it in directory names."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    """I replace an archive only after a complete temporary write succeeds."""

    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        np.savez_compressed(temporary_path, **arrays)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _atomic_write_text(path: Path, content: str) -> None:
    """I atomically replace JSON provenance after serializing it completely."""

    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        suffix=".tmp",
        delete=False,
        encoding="utf-8",
        mode="w",
    ) as temporary:
        temporary.write(content)
        temporary_path = Path(temporary.name)
    try:
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
