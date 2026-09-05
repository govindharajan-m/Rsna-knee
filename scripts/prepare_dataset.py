"""I build the MRI preprocessing cache and Parquet manifest from a configurable data root."""

from __future__ import annotations

import argparse
import sys
from importlib import import_module
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

audit_module = import_module("kneescope12.data.audit")
prepare_module = import_module("kneescope12.data.prepare")
config_module = import_module("kneescope12.utils.config")
logging_module = import_module("kneescope12.utils.logging")


def main() -> int:
    """I run cached preprocessing without assuming a local competition-data path."""

    parser = argparse.ArgumentParser(description="Reconstruct and cache MRI series from DICOM metadata.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "baseline.yaml")
    parser.add_argument("--data-root", "--dataset-root", dest="data_root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "experiments")
    parser.add_argument("--qc-samples", type=int, default=0)
    parser.add_argument("--max-series", type=int)
    args = parser.parse_args()
    config = config_module.resolve_runtime_paths(
        config_module.load_config(args.config), project_root=PROJECT_ROOT
    )
    paths = config.get("paths", {})
    logging_module.configure_logging(config.get("logging", {}).get("level", "INFO"))
    preprocessing = prepare_module.PreprocessingConfig.from_mapping(config.get("preprocessing", {}))
    audit_config = audit_module.DatasetAuditConfig.from_mapping(config.get("audit"))
    cache_root = args.cache_root or config_module.resolve_path(paths["cache_root"], project_root=PROJECT_ROOT)
    manifest_path = args.manifest_path or config_module.resolve_path(
        paths["manifest_output"], project_root=PROJECT_ROOT
    )
    result = prepare_module.prepare_dataset(
        args.data_root,
        cache_root=cache_root,
        manifest_path=manifest_path,
        output_dir=args.output_dir,
        preprocessing=preprocessing,
        audit_config=audit_config,
        qc_samples=args.qc_samples,
        max_series=args.max_series,
    )
    print(
        "Prepared "
        f"{result.processed_series_count} series; skipped {result.skipped_series_count}; "
        f"cache hits {result.cache_hit_count}. Manifest: {result.manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
