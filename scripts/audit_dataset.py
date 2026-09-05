"""I expose the metadata-only dataset audit as a direct Python script."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

main = import_module("kneescope12.data.audit").main


if __name__ == "__main__":
    raise SystemExit(main())
