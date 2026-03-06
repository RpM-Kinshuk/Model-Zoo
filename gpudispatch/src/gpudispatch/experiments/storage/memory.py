"""In-memory storage backend for experiments.

This module provides a simple in-memory storage implementation that stores
all data in Python dictionaries. Data is not persisted between process restarts.
Useful for testing and development.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from gpudispatch.experiments.storage.base import Storage
from gpudispatch.experiments.trial import Trial


class MemoryStorage(Storage):
    """In-memory storage backend.

    Stores all experiment data in Python dictionaries. Data is lost when
    the process exits or the storage instance is garbage collected.

    This backend is useful for:
    - Unit testing
    - Development and prototyping
    - Short-lived experiments that don't need persistence

    Example:
        >>> storage = MemoryStorage()
        >>> storage.save_trial("exp1", Trial(id=1, params={"lr": 0.01}))
        >>> trial = storage.load_trial("exp1", 1)
        >>> trial.params
        {'lr': 0.01}
    """

    def __init__(self) -> None:
        """Initialize empty in-memory storage."""
        # Structure: {experiment_name: {trial_id: Trial}}
        self._trials: Dict[str, Dict[int, Trial]] = {}
        # Structure: {experiment_name: config_dict}
        self._configs: Dict[str, Dict[str, Any]] = {}

    def save_trial(self, experiment_name: str, trial: Trial) -> None:
        """Save a trial to memory.

        Args:
            experiment_name: Name of the experiment.
            trial: Trial to save.
        """
        if experiment_name not in self._trials:
            self._trials[experiment_name] = {}
        self._trials[experiment_name][trial.id] = trial

    def load_trial(self, experiment_name: str, trial_id: int) -> Optional[Trial]:
        """Load a single trial by ID.

        Args:
            experiment_name: Name of the experiment.
            trial_id: ID of the trial to load.

        Returns:
            The trial if found, None otherwise.
        """
        if experiment_name not in self._trials:
            return None
        return self._trials[experiment_name].get(trial_id)

    def load_trials(self, experiment_name: str) -> List[Trial]:
        """Load all trials for an experiment.

        Args:
            experiment_name: Name of the experiment.

        Returns:
            List of all trials for the experiment.
        """
        if experiment_name not in self._trials:
            return []
        return list(self._trials[experiment_name].values())

    def save_config(self, experiment_name: str, config: Dict[str, Any]) -> None:
        """Save experiment configuration.

        Args:
            experiment_name: Name of the experiment.
            config: Configuration dictionary to save.
        """
        self._configs[experiment_name] = config

    def load_config(self, experiment_name: str) -> Optional[Dict[str, Any]]:
        """Load experiment configuration.

        Args:
            experiment_name: Name of the experiment.

        Returns:
            Configuration dictionary if found, None otherwise.
        """
        return self._configs.get(experiment_name)

    def list_experiments(self) -> List[str]:
        """List all experiment names in storage.

        Returns:
            List of experiment names that have either trials or configs.
        """
        experiments = set(self._trials.keys()) | set(self._configs.keys())
        return list(experiments)
