"""Minimal experiment registry for managing experiments.

This module provides a simple registry for discovering and loading
experiments stored on disk. It uses a global default storage location
that can be configured.

Example:
    >>> from gpudispatch.experiments.registry import (
    ...     set_experiment_dir,
    ...     list_experiments,
    ...     load,
    ... )
    >>>
    >>> # Set custom experiment directory
    >>> set_experiment_dir("./my_experiments")
    >>>
    >>> # List all experiments
    >>> experiments = list_experiments()
    >>> print(experiments)  # ["exp1", "exp2", ...]
    >>>
    >>> # Load a specific experiment
    >>> exp = load("exp1")
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from gpudispatch.experiments.storage import FileStorage, Storage

if TYPE_CHECKING:
    from gpudispatch.experiments.experiment import Experiment


# Global default storage location
_default_storage: Optional[Storage] = None


def set_experiment_dir(path: str) -> None:
    """Set default experiment directory.

    Creates a FileStorage backend pointing to the specified directory.
    The directory will be created if it doesn't exist.

    Args:
        path: Path to the experiment directory.
    """
    global _default_storage
    _default_storage = FileStorage(Path(path))


def get_storage() -> Storage:
    """Get default storage.

    Returns the configured storage, or creates a FileStorage at
    ~/.gpudispatch/experiments if not configured.

    Returns:
        The default Storage instance.
    """
    global _default_storage
    if _default_storage is None:
        default_dir = Path.home() / ".gpudispatch" / "experiments"
        _default_storage = FileStorage(default_dir)
    return _default_storage


def list_experiments() -> List[str]:
    """List all experiment names.

    Returns a list of experiment names found in the default storage.

    Returns:
        List of experiment names.
    """
    return get_storage().list_experiments()


def load(name: str) -> Optional["Experiment"]:
    """Load experiment by name.

    Loads an experiment from the default storage.

    Args:
        name: Name of the experiment to load.

    Returns:
        The Experiment if found, None otherwise.
    """
    # Import here to avoid circular import
    from gpudispatch.experiments.experiment import Experiment

    return Experiment.load(name, get_storage())


def _reset_storage() -> None:
    """Reset the global storage (for testing).

    This is an internal function used by tests to reset the global
    storage state between tests.
    """
    global _default_storage
    _default_storage = None
