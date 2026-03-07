#!/usr/bin/env python3
"""Utility wrapper for model-list creation commands.

This keeps the public utility path (`utils/create_model_list.py`) stable while
reusing the current implementation under `examples/create_model_list.py`.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_impl() -> ModuleType:
    """Load the canonical create_model_list implementation module."""
    impl_path = Path(__file__).resolve().parent.parent / "examples" / "create_model_list.py"
    spec = importlib.util.spec_from_file_location("esd_experiment_create_model_list_impl", impl_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load create_model_list implementation from {impl_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    impl = _load_impl()
    impl.main()


if __name__ == "__main__":
    main()
