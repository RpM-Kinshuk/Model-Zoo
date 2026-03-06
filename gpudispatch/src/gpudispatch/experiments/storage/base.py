"""Abstract base class for experiment storage backends.

This module defines the Storage ABC that all storage backends must implement.
Storage backends are responsible for persisting experiment trials and configuration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from gpudispatch.experiments.trial import Trial


class Storage(ABC):
    """Abstract base class for experiment storage.

    Storage backends provide persistence for experiment data including:
    - Trials with parameters, metrics, and status
    - Experiment configuration

    Implementations must be able to store and retrieve trials by experiment name
    and trial ID, and maintain isolation between different experiments.

    Example:
        >>> class MyStorage(Storage):
        ...     def save_trial(self, experiment_name, trial): ...
        ...     # ... implement other methods
        >>> storage = MyStorage()
        >>> storage.save_trial("exp1", trial)
        >>> loaded = storage.load_trial("exp1", trial.id)
    """

    @abstractmethod
    def save_trial(self, experiment_name: str, trial: Trial) -> None:
        """Save a trial to storage.

        If a trial with the same ID already exists for this experiment,
        it should be overwritten.

        Args:
            experiment_name: Name of the experiment.
            trial: Trial to save.
        """
        ...

    @abstractmethod
    def load_trial(self, experiment_name: str, trial_id: int) -> Optional[Trial]:
        """Load a single trial by ID.

        Args:
            experiment_name: Name of the experiment.
            trial_id: ID of the trial to load.

        Returns:
            The trial if found, None otherwise.
        """
        ...

    @abstractmethod
    def load_trials(self, experiment_name: str) -> List[Trial]:
        """Load all trials for an experiment.

        Args:
            experiment_name: Name of the experiment.

        Returns:
            List of all trials for the experiment. Empty list if experiment
            doesn't exist or has no trials.
        """
        ...

    @abstractmethod
    def save_config(self, experiment_name: str, config: Dict[str, Any]) -> None:
        """Save experiment configuration.

        If a config already exists for this experiment, it should be overwritten.

        Args:
            experiment_name: Name of the experiment.
            config: Configuration dictionary to save.
        """
        ...

    @abstractmethod
    def load_config(self, experiment_name: str) -> Optional[Dict[str, Any]]:
        """Load experiment configuration.

        Args:
            experiment_name: Name of the experiment.

        Returns:
            Configuration dictionary if found, None otherwise.
        """
        ...

    @abstractmethod
    def list_experiments(self) -> List[str]:
        """List all experiment names in storage.

        Returns:
            List of experiment names that have either trials or configs stored.
        """
        ...
