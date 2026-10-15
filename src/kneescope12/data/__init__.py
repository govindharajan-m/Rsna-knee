"""I expose data interfaces without importing unrelated audit dependencies."""

from importlib import import_module
from typing import Any

__all__ = [
    "DatasetAuditConfig",
    "DatasetAuditor",
    "DicomImageReadResult",
    "DicomMetadata",
    "DicomReadResult",
    "SliceOrdering",
    "order_slices",
    "read_dicom_image",
    "read_dicom_metadata",
]

_EXPORT_MODULES = {
    "DatasetAuditConfig": "audit",
    "DatasetAuditor": "audit",
    "DicomImageReadResult": "dicom",
    "DicomMetadata": "dicom",
    "DicomReadResult": "dicom",
    "read_dicom_image": "dicom",
    "read_dicom_metadata": "dicom",
    "SliceOrdering": "ordering",
    "order_slices": "ordering",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f".{module_name}", __name__), name)
