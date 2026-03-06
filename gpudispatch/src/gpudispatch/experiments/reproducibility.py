"""Minimal reproducibility utilities for experiments.

This module provides utilities for capturing reproducibility context
and setting random seeds to ensure experiments can be reproduced.

Example:
    >>> from gpudispatch.experiments.reproducibility import capture_context, set_seeds
    >>>
    >>> # Set seeds before running experiment
    >>> set_seeds(42)
    >>>
    >>> # Capture context for logging
    >>> context = capture_context(seed=42)
    >>> print(context["python_version"])
    >>> print(context["git_commit"])
"""

from __future__ import annotations

import random
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, Optional


def capture_context(seed: Optional[int] = None) -> Dict[str, Any]:
    """Capture reproducibility context.

    Returns a dictionary containing information needed to reproduce
    an experiment run, including Python version, timestamp, git commit,
    and random seed.

    Args:
        seed: Optional random seed that was used for the experiment.

    Returns:
        Dictionary with reproducibility context:
        - python_version: Full Python version string
        - timestamp: ISO format timestamp of capture
        - git_commit: Current git commit hash, or None if not in git repo
        - random_seed: The seed value if provided, otherwise None
    """
    return {
        "python_version": sys.version,
        "timestamp": datetime.now().isoformat(),
        "git_commit": _get_git_commit(),
        "random_seed": seed,
    }


def _get_git_commit() -> Optional[str]:
    """Get current git commit hash, or None.

    Returns:
        The current git commit hash as a string, or None if:
        - Not in a git repository
        - Git is not installed
        - Any other error occurs
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility.

    Sets the random seed for Python's random module, and optionally
    for numpy and torch if they are available.

    Args:
        seed: Integer seed value to use.
    """
    # Set Python random seed
    random.seed(seed)

    # Set numpy seed if available
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    # Set torch seed if available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
