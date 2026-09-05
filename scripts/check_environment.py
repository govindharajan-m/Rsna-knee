"""I verify that the cloud environment can run the KneeScope-12 data pipeline."""

from __future__ import annotations

import math
import os
import shutil
import socket
import sys
from pathlib import Path

import psutil

REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pydicom": "pydicom",
    "yaml": "yaml",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "pyarrow": "pyarrow",
    "langdetect": "langdetect",
    "torch": "torch",
    "psutil": "psutil",
}


def _require_module(name: str, package_name: str | None = None) -> object:
    try:
        module = __import__(name)
    except ImportError as exc:  # pragma: no cover - CLI guard
        raise RuntimeError(f"Missing required package: {package_name or name}") from exc
    return module


def _bytes_to_gib(value: float) -> float:
    return value / (1024**3)


def _gpu_info() -> tuple[str | None, str | None, float | None]:
    try:
        import torch

        if not torch.cuda.is_available():
            return None, None, None
        count = torch.cuda.device_count()
        if count == 0:
            return None, None, None
        name = torch.cuda.get_device_name(0)
        total_memory_bytes = torch.cuda.get_device_properties(0).total_memory
        return name, f"{torch.version.cuda or 'unknown'}", _bytes_to_gib(total_memory_bytes)
    except Exception:
        return None, None, None


def _disk_space(path: str) -> tuple[float, float]:
    usage = shutil.disk_usage(path)
    return _bytes_to_gib(usage.total), _bytes_to_gib(usage.free)


def main() -> int:
    print("KneeScope-12 environment check")
    print("=" * 32)
    print(f"Python: {sys.version.replace(chr(10), ' ')}")
    print(f"Platform: {sys.platform}")
    print(f"Hostname: {socket.gethostname()}")

    try:
        import torch

        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA version: {torch.version.cuda or 'N/A'}")
    except Exception as exc:  # pragma: no cover - CLI guard
        print(f"PyTorch: MISSING ({exc})")
        raise SystemExit(1)

    gpu_name, cuda_version, gpu_memory = _gpu_info()
    print(f"GPU name: {gpu_name or 'N/A'}")
    print(f"GPU memory (GiB): {gpu_memory:.2f}" if gpu_memory is not None else "GPU memory (GiB): N/A")
    if gpu_name is None:
        print("Warning: CUDA runtime is unavailable or no NVIDIA GPU is visible.")

    memory = psutil.virtual_memory()
    print(f"CPU count: {psutil.cpu_count(logical=True) or 'N/A'}")
    print(f"RAM: {memory.total / (1024**3):.2f} GiB")
    for package_name, import_name in REQUIRED_PACKAGES.items():
        try:
            module = _require_module(import_name, package_name)
            version_name = getattr(module, "__version__", "unknown")
            print(f"{package_name}: {version_name}")
        except RuntimeError as exc:
            print(f"Missing dependency: {exc}")
            raise SystemExit(1)

    data_root = os.getenv("KNEESCOPE_DATA_ROOT", ".")
    total_disk, free_disk = _disk_space(str(Path(data_root).resolve() if Path(data_root).exists() else Path.cwd()))
    print(f"Disk total: {total_disk:.2f} GiB")
    print(f"Disk free: {free_disk:.2f} GiB")
    print("Environment check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
