"""Checkpoint evidence extraction without constructing models."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT, _ROOT / "esd_experiment" / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

EVIDENCE_VERSION = "3"
READABLE_EVIDENCE_VERSIONS = frozenset({"1", "2", EVIDENCE_VERSION})
